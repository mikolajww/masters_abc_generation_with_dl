import datetime
import os
import pickle
import pprint
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from torch import Tensor
from einops import rearrange, reduce, repeat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import utils
from dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from utils import setup_matplotlib_style, pad_collate


class GRUSymetricalVAE(torch.nn.Module):
    def __init__(self,
                 vocab,
                 embedding_size,
                 encoder_decoder_hidden_size,
                 encoder_decoder_num_layers,
                 latent_vector_size,
                 dropout_prob
                 ):
        super(GRUSymetricalVAE, self).__init__()
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.encoder_decoder_hidden_size = encoder_decoder_hidden_size
        self.encoder_decoder_num_layers = encoder_decoder_num_layers
        self.vocab_size = len(vocab)
        self.latent_vector_size = latent_vector_size
        self.dropout_prob = dropout_prob

        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_size)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_prob)
        self.encoder = torch.nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.encoder_decoder_hidden_size,
            num_layers=self.encoder_decoder_num_layers,
            batch_first=True
        )

        self.flattened_enc_size = self.flattened_enc_size = self.encoder_decoder_hidden_size * self.encoder_decoder_num_layers

        self.enc_hidden_to_mean = torch.nn.Linear(self.flattened_enc_size, latent_vector_size)
        self.enc_hidden_to_logv = torch.nn.Linear(self.flattened_enc_size, latent_vector_size)

        self.latent_to_dec_hidden = torch.nn.Linear(self.latent_vector_size, self.flattened_enc_size)

        self.decoder = torch.nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.encoder_decoder_hidden_size,
            num_layers=self.encoder_decoder_num_layers,
            batch_first=True
        )
        self.dec_output_to_vocab = torch.nn.Linear(self.encoder_decoder_hidden_size, self.vocab_size)

    def forward(self, X, len_X):
        # input is integer encoded and padded with <PAD> token to max_len
        batch_size, padded_sequence_len = X.size()
        X = self.embedding(X)
        # X = (batch size, max_len, embedding_size)
        # Pack
        packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

        _, latent_hidden = self.encoder(packed_X)
        # latent_hidden = (num_layers, batch_size, hidden_size)
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#torch.nn.GRU
        latent_hidden = rearrange(latent_hidden, "l b h -> b (l h)")

        mean = self.enc_hidden_to_mean(latent_hidden)
        log_variance = self.enc_hidden_to_logv(latent_hidden)
        std_dev = torch.exp(0.5 * log_variance)

        # TODO Readup more on that
        z = torch.randn((batch_size, self.latent_vector_size), device=DEVICE)
        z = mean + z * std_dev

        latent_hidden = self.latent_to_dec_hidden(z)
        latent_hidden = rearrange(
            latent_hidden, "b (l h) -> l b h ",
            b=batch_size, h=self.encoder_decoder_hidden_size
        )

        X = self.emb_dropout(X)
        # is this necessary?
        packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

        output, _ = self.decoder(packed_X, latent_hidden)

        # unpack
        out_X, out_len_X = pad_packed_sequence(output, batch_first=True)

        out_X = self.dec_output_to_vocab(out_X)
        # Log softmax returns large negative numbers for low probas and near-zero for high probas
        # last dim is actual embeddings
        out_X = F.log_softmax(out_X, dim=-1)

        return out_X, out_len_X, mean, log_variance, z

    def generate(self, latent_z, sos_idx, unk_idx):
        # latent z should be of shape (batch_size, latent_size)

        with torch.no_grad():
            latent_hidden = self.latent_to_dec_hidden(latent_z)
            latent_hidden = rearrange(
                latent_hidden, "b (l h) -> l b h",
                l=self.encoder_decoder_num_layers,
                h=self.encoder_decoder_hidden_size
            )
            X = torch.zeros(CONFIG["max_len"], device=DEVICE).fill_(unk_idx).long()
            X[0] = sos_idx
            len_X = torch.tensor([len(X)]).long()
            X = rearrange(X, "x -> 1 x")
            X = self.embedding(X)
            packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

            output, _ = self.decoder(packed_X, latent_hidden)

            # unpack
            out_X, out_len_X = pad_packed_sequence(output, batch_first=True)

            out_X = self.dec_output_to_vocab(out_X)
            # Log softmax returns large negative numbers for low probas and near-zero for high probas
            # last dim is actual embeddings
            out_X = F.log_softmax(out_X, dim=-1)
            tune = self.vocab.lookup_tokens(torch.argmax(out_X, dim=-1).cpu().squeeze().tolist())
            return tune

    def generate_autoregressively(self, latent_z, bos_idx, eos_idx):
        attempt = 0
        tries = []
        while True:
            attempt += 1
            latent_hidden = self.latent_to_dec_hidden(latent_z)
            latent_hidden = rearrange(
                latent_hidden, "b (l h) -> l b h",
                l=self.encoder_decoder_num_layers,
                h=self.encoder_decoder_hidden_size
            )

            generated_tune = []
            X = torch.tensor([bos_idx], device=DEVICE).long()

            while X.squeeze().item() != eos_idx:
                with torch.no_grad():
                    X = rearrange(X, "x -> 1 x")
                    X = self.embedding(X)
                    output, latent_hidden = self.decoder(X, latent_hidden)

                    out_X = self.dec_output_to_vocab(output)
                    # Log softmax returns large negative numbers for low probas and near-zero for high probas
                    # last dim is actual embeddings
                    probas = F.softmax(out_X, dim=-1).cpu().squeeze().numpy()

                    chosen_index = np.random.choice(len(out_X.squeeze()), p=probas)
                    X = torch.tensor([chosen_index], device=DEVICE).long()
                    # X = torch.argmax(out_X, dim=-1)

                    generated_tune.extend(self.vocab.lookup_tokens([chosen_index]))

            tune = "".join(generated_tune).replace("<EOS>", "").replace("<BOS>","").strip()
            if utils.is_valid_abc(tune):
                break
            else:
                tries.append(tune)
        return tune, tries, attempt


def padded_kl_nll_loss(predictions, len_predictions,
                       targets, len_targets,
                       mean, log_variance,
                       optimizer_step):
    # KL Annealing https://arxiv.org/pdf/1511.06349.pdf
    nll_loss = torch.tensor(0.0, device=DEVICE)

    for i in range(predictions.size(0)):
        nll = F.nll_loss(
            predictions[i][:len_predictions[i]],
            targets[i][:len_targets[i]],
            reduction="sum",
            ignore_index=0
        )
        nll = nll / len_predictions[i].cuda()
        nll_loss += nll

    # batch_loss = batch_loss / predictions.size(0)

    kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    k, step, x0 = 0.0025, optimizer_step, 2500
    kl_weight = float(1 / (1 + np.exp(-k * (step - x0))))

    # ELBO Loss
    l = 1 if predictions.size(0) == 0 else predictions.size(0)
    loss = (nll_loss + kl_div * kl_weight) / l
    return loss, (nll_loss, kl_div, kl_weight)


def train():
    dataset = ABCInMemoryDataset(CONFIG["path_to_abc"], max_len=CONFIG["max_len"], min_len=CONFIG["min_len"], cut_or_filter="cut")

    train_data_loader, valid_data_loader, test_data_loader = split_train_valid_test_dataloaders(
        dataset, train_percent=0.8, valid_percent=0.1,
        batch_size=CONFIG["batch_size"],
        collate_fn=pad_collate)

    model = GRUSymetricalVAE(
        vocab=dataset.vocabulary,
        embedding_size=CONFIG["embedding_size"],
        encoder_decoder_hidden_size=CONFIG["encoder_decoder_hidden_size"],
        encoder_decoder_num_layers=CONFIG["encoder_decoder_num_layers"],
        latent_vector_size=CONFIG["latent_vector_size"],
        dropout_prob=CONFIG["dropout_prob"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08
    )

    history = {
        "train": {
            "loss": []
        },
        "val": {
            "loss": []
        }
    }
    epoch_times = []
    model.train()
    optimizer_step = 0
    n_train_batches = len(train_data_loader)
    # train/val, epoch, batch, (nll_loss, kl_div, kl_weight)
    kl_history = torch.tensor(np.zeros((2, CONFIG["epochs"], n_train_batches, 3)), device=DEVICE)
    for epoch in range(CONFIG["epochs"]):
        torch.cuda.empty_cache()
        epoch_start_time = time.perf_counter()

        minibatch_losses = torch.tensor(np.zeros(n_train_batches), device=DEVICE)

        for i, (batch_X, batch_y, len_X, len_y) in enumerate(tqdm.tqdm(train_data_loader)):
            # this is equivalent to optimizer.zero_grad()
            # reset gradients every minibatch!
            # set_to_none supposedly is more efficient as per
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)
            predictions, len_predictions, mean, log_variance, z = model(batch_X, len_X)
            targets = batch_y.long()
            loss, (nll_loss, kl_div, kl_weight) = padded_kl_nll_loss(
                predictions, len_predictions, targets, len_y, mean, log_variance, optimizer_step
            )
            kl_history[0, epoch, i] = torch.tensor([nll_loss, kl_div, kl_weight], device=DEVICE)
            minibatch_losses[i] = loss

            loss.backward()
            optimizer.step()
            optimizer_step += 1

            if i % 50 == 0:
                tqdm.tqdm.write(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {loss.item()}")

        history["train"]["loss"].append(torch.nanmean(minibatch_losses).item())

        with torch.no_grad():
            n_batches = len(valid_data_loader)
            val_minibatch_losses = torch.tensor(np.zeros(n_batches), device=DEVICE)
            for i, (batch_X, batch_y, len_X, len_y) in enumerate(valid_data_loader):
                predictions, len_predictions, mean, log_variance, z = model(batch_X, len_X)
                targets = batch_y.long()
                val_loss, (nll_loss, kl_div, kl_weight) = padded_kl_nll_loss(
                    predictions, len_predictions, targets, len_y, mean, log_variance, optimizer_step
                )
                val_minibatch_losses[i] = val_loss

        history["val"]["loss"].append(torch.nanmean(val_minibatch_losses).item())
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_time)
        tqdm.tqdm.write(
            f"Epoch {epoch + 1}: [{epoch_time:.2f}s] Train loss: {history['train']['loss'][epoch]} | Val loss: {history['val']['loss'][epoch]}")

        scheduler.step(val_loss)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "CONFIG": CONFIG,
        "embedding_size": model.embedding_size,
        "encoder_decoder_hidden_size": model.encoder_decoder_hidden_size,
        "encoder_decoder_num_layers": model.encoder_decoder_num_layers,
        "latent_vector_size": model.latent_vector_size,
        "dropout_prob": model.dropout_prob,
        "vocab": model.vocab,
        "model": model
    }

    folder_name = f"experiment_{model.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(folder_name)
    save_filename = f"{model.__class__.__name__}.pth"
    torch.save(state, f"{folder_name}/{save_filename}")
    pickle.dump(history, open(f"{folder_name}/history.pkl", "wb"))
    np.save(f"{folder_name}/kl_history.npy", kl_history.cpu().numpy())

    with open(f"{folder_name}/params.txt", "w") as f:
        f.writelines(pprint.pformat(CONFIG))
    with open(f"{folder_name}/time.txt", "w") as f:
        f.writelines(str(epoch_times))

    with open(f"{folder_name}/experimet_summary.txt", "a") as f:
        f.writelines(folder_name + "\n")
        f.writelines(pprint.pformat(CONFIG) + "\n")
        f.write(f"Time : {np.array(epoch_times).mean() :.2f}\n")
        f.write("\n\n")

    print(f"Experiment saved to {folder_name}")


def evaluate(model_path):
    print(f"Loading model {model_path}")
    state = torch.load(model_path)
    CONFIG = state["CONFIG"]
    model = state["model"]
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    bos_idx = model.vocab.lookup_indices(["<BOS>"])[0]
    unk_idx = model.vocab.lookup_indices(["<UNK>"])[0]
    eos_idx = model.vocab.lookup_indices(["<EOS>"])[0]


    TRIES = 500
    print(model)
    print(f"Evaluating model for {TRIES} tries")
    base_folder = Path(model_path).parent
    tunes_out_folder = base_folder.joinpath("output_tunes")
    Path.mkdir(tunes_out_folder, exist_ok=True)

    eval_start = time.perf_counter()
    eval_times = []
    n_of_attempts_list = []
    for i in tqdm.trange(TRIES):
        eval_attempt_start = time.perf_counter()

        latent_z = torch.randn(model.latent_vector_size, device=DEVICE)
        latent_z = rearrange(latent_z, "z -> 1 z")
        generated_tune, tries, n_of_attempts = model.generate_autoregressively(latent_z, bos_idx, eos_idx)

        eval_times.append(time.perf_counter() - eval_attempt_start)
        with open(f"{tunes_out_folder}/tune_{i}_correct_attempts_{n_of_attempts}.abc", "w") as out:
            out.writelines(generated_tune)
        with open(f"{tunes_out_folder}/tune_{i}_incorrect_attempts_{n_of_attempts}.abc", "w") as out:
            out.writelines(tries)
        n_of_attempts_list.append(n_of_attempts)

    eval_end = time.perf_counter()

    with open(f"{base_folder}/n_of_tries.txt", "w") as f:
        f.writelines(str(n_of_attempts_list))

    with open(f"{base_folder}/eval_times.pkl", "wb") as f:
        pickle.dump(eval_times, f)

    eval_times = np.array(eval_times)

    hist = pickle.load(open(base_folder.joinpath("history.pkl"), "rb"))
    plt.plot(np.arange(1, len(hist["train"]["loss"]) + 1), hist["train"]["loss"], label="Training loss")
    plt.plot(np.arange(1, len(hist["val"]["loss"]) + 1), hist["val"]["loss"], label="Validation loss")
    plt.legend()
    plt.title(
        f"Minimum traning loss: {np.array(hist['train']['loss']).min() :.5f}\nMinimum validation loss: {np.array(hist['val']['loss']).min():.5f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Crossentropy)")
    plt.savefig(f"{Path(model_path).parent}/training_val_losses.png")
    plt.show()

    eval_summary_str = f"Took {eval_end - eval_start:.2f}s to evaluate. [min = {eval_times.min()}, avg = {eval_times.mean()}, max = {eval_times.max()}]"
    with open(f"{base_folder}/model_summary.txt", "w") as f:
        f.writelines(pprint.pformat(CONFIG))
        f.write("\n")
        f.writelines(str(model))
        f.writelines(eval_summary_str)

    print(eval_summary_str)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
    ("path_to_abc", "../data_processed/abc_parsed_cleanup5.abc"),
    ("batch_size", 64),
    ("lr", 0.001),
    ("embedding_size", 32),
    ("latent_vector_size", 256),
    ("encoder_decoder_hidden_size", 256),
    ("encoder_decoder_num_layers", 3),
    ("dropout_prob", 0.4),
    ("epochs", 10),
    ("cut_or_filter", "cut"),
    ("max_len", 700),
    ("min_len", 30)
])

if __name__ == "__main__":
    setup_matplotlib_style()
    # mode = "train"
    mode = "eval"
    # BEST MODEL BELOW
    model_path = "experiment_GRUSymetricalVAE_20220901-015105-to-test/GRUSymetricalVAE.pth"
    # model_path = "experiment_GRUSymetricalVAE_20220723-022447/GRUSymetricalVAE.pth"
    if mode == "train":
        train()
    elif mode == "eval":
        evaluate(model_path)
