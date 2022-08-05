import datetime
import os
import pickle
import pprint
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from utils import setup_matplotlib_style, plot_training_history_from_pickle, pad_collate


class RVAE_Seq2Seq(torch.nn.Module):
    def __init__(
            self,
            vocab,
            embedding_dim,
            enc_dec_hidden,
            enc_dec_nlayers,
            latent_dim,
            dropout_prob
    ):
        super(RVAE_Seq2Seq, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.enc_dec_hidden = enc_dec_hidden
        self.enc_dec_nlayers = enc_dec_nlayers
        self.vocab_size = len(vocab)
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_prob)
        self.encoder = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.enc_dec_hidden,
            num_layers=self.enc_dec_nlayers,
            batch_first=True
        )

        self.enc_hidden_to_mean = torch.nn.Linear(
            in_features=self.enc_dec_hidden * self.enc_dec_nlayers,
            out_features=self.latent_dim
        )
        self.enc_hidden_to_logv = torch.nn.Linear(
            in_features=self.enc_dec_hidden * self.enc_dec_nlayers,
            out_features=self.latent_dim
        )

        self.latent_to_dec_input = torch.nn.Linear(
            in_features=self.latent_dim,
            out_features=self.embedding_dim
        )

        self.decoder = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.enc_dec_hidden,
            num_layers=self.enc_dec_nlayers,
            batch_first=True
        )
        self.dec_output_to_vocab = torch.nn.Linear(self.enc_dec_hidden, self.vocab_size)

    def forward(self, X, len_X):
        batch_size, padded_sequence_len = X.size()
        X = self.embedding(X)
        packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

        _, latent_hidden = self.encoder(packed_X)
        latent_hidden = rearrange(latent_hidden, "l b h -> b (l h)")

        mean = self.enc_hidden_to_mean(latent_hidden)
        log_variance = self.enc_hidden_to_logv(latent_hidden)
        std_dev = torch.exp(0.5 * log_variance)
        z = torch.randn((batch_size, self.latent_dim), device=DEVICE)
        z = mean + z * std_dev

        decoder_X = self.latent_to_dec_input(z)
        decoder_X = repeat(decoder_X, "b z -> b repeat z", repeat=padded_sequence_len)
        packed_X = pack_padded_sequence(decoder_X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

        output, _ = self.decoder(packed_X)

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
                l=self.enc_dec_nlayers,
                h=self.enc_dec_hidden
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
            return "".join(tune)

    def generate_tok_by_tok(self, latent_z, bos_idx, eos_idx):

        latent_hidden = self.latent_to_dec_hidden(latent_z)
        latent_hidden = rearrange(
            latent_hidden, "b (l h) -> l b h",
            l=self.enc_dec_nlayers,
            h=self.enc_dec_hidden
        )

        generated_tune = []
        X = torch.tensor([bos_idx], device=DEVICE).long()

        i = 0
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
                i += 1
                if i >= 1000:
                    return "".join(generated_tune)

        return "".join(generated_tune).replace("<EOS>", "")


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

    l = 1 if predictions.size(0) == 0 else predictions.size(0)
    nll_loss = nll_loss / l

    # batch_loss = batch_loss / predictions.size(0)
    kl_div = (-0.5 * torch.sum(log_variance - torch.pow(mean, 2) - torch.exp(log_variance) + 1, dim=1)).mean()
    k, step, x0 = 0.0025, optimizer_step, 2500
    kl_weight = 1  # float(1 / (1 + np.exp(-k * (step - x0))))

    # ELBO Loss


    loss = (nll_loss + kl_div * kl_weight)
    return loss, (nll_loss, kl_div, kl_weight)


def train():
    dataset = ABCInMemoryDataset(CONFIG["path_to_abc"], max_len=CONFIG["max_len"], cut_or_filter="cut")

    collate = partial(pad_collate, device=DEVICE, pad_idx=dataset.PAD_IDX)

    train_data_loader, valid_data_loader, test_data_loader = split_train_valid_test_dataloaders(
        dataset, train_percent=0.8, valid_percent=0.1,
        batch_size=CONFIG["batch_size"],
        collate_fn=collate)

    model = RVAE_Seq2Seq(
        vocab=dataset.vocabulary,
        embedding_dim=CONFIG["embedding_size"],
        enc_dec_hidden=CONFIG["encoder_decoder_hidden_size"],
        enc_dec_nlayers=CONFIG["encoder_decoder_num_layers"],
        latent_dim=CONFIG["latent_vector_size"],
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
            # Train as Autoencoder
            loss, (nll_loss, kl_div, kl_weight) = padded_kl_nll_loss(
                predictions, len_predictions, batch_X.long(), len_X, mean, log_variance, optimizer_step
            )
            kl_history[0, epoch, i] = torch.tensor([nll_loss, kl_div, kl_weight], device=DEVICE)
            minibatch_losses[i] = loss

            loss.backward()
            optimizer.step()
            optimizer_step += 1

            if i % 50 == 0:
                tqdm.tqdm.write(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {loss.item()} [nll_loss = {nll_loss.item()}, kl_div = {kl_div.item()}, kl_weight = {kl_weight}]")

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
        "CONFIG": CONFIG,
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

    eos_tok = "<EOS>"
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
    for i in tqdm.trange(1):
        eval_attempt_start = time.perf_counter()

        latent_z = torch.randn(model.latent_dim, device=DEVICE)
        latent_z = rearrange(latent_z, "z -> 1 z")
        # generated_tune, tries, n_of_attempts
        generated_tune = model.generate(latent_z, bos_idx, unk_idx)
        # generated_tune = #model.generate_tok_by_tok(latent_z, bos_idx, eos_idx) #
        print(generated_tune)

    #     eval_times.append(time.perf_counter() - eval_attempt_start)
    #     with open(f"{tunes_out_folder}/tune_{i}_correct_attempts_{n_of_attempts}.abc", "w") as out:
    #         out.writelines(generated_tune)
    #     with open(f"{tunes_out_folder}/tune_{i}_incorrect_attempts_{n_of_attempts}.abc", "w") as out:
    #         out.writelines(tries)
    #     n_of_attempts_list.append(n_of_attempts)
    #
    # eval_end = time.perf_counter()
    #
    # with open(f"{base_folder}/n_of_tries.txt", "w") as f:
    #     f.writelines(str(n_of_attempts_list))
    #
    # with open(f"{base_folder}/eval_times.pkl", "wb") as f:
    #     pickle.dump(eval_times, f)
    #
    # eval_times = np.array(eval_times)
    #
    plot_training_history_from_pickle(base_folder, "history.pkl")

    #
    # eval_summary_str = f"Took {eval_end - eval_start:.2f}s to evaluate. [min = {eval_times.min()}, avg = {eval_times.mean()}, max = {eval_times.max()}]"
    # with open(f"{base_folder}/model_summary.txt", "w") as f:
    #     f.writelines(pprint.pformat(CONFIG))
    #     f.write("\n")
    #     f.writelines(str(model))
    #     f.writelines(eval_summary_str)
    #
    # print(eval_summary_str)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
    ("path_to_abc", "../data_processed/abc_parsed_cleanup2.abc"),
    ("batch_size", 64),
    ("lr", 0.01),
    ("embedding_size", 32),
    ("latent_vector_size", 256),
    ("encoder_decoder_hidden_size", 256),
    ("encoder_decoder_num_layers", 2),
    ("dropout_prob", 0.3),
    ("epochs", 10),
    ("cut_or_filter", "filter"),
    ("max_len", 700)
])

if __name__ == "__main__":
    setup_matplotlib_style()
    mode = "train"
    # mode = "eval"
    model_path = "experiment_GRUSymetricalVAE_20220727-094750/GRUSymetricalVAE.pth"
    if mode == "train":
        train()
    elif mode == "eval":
        evaluate(model_path)
