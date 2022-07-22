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
from torch import Tensor
from einops import rearrange, reduce, repeat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from abc_magisterka_kod.dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from utils import setup_matplotlib_style


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

        # TODO replace some tokens with unk at random?

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

    def generate(self, latent_z):
        # latent z should be of shape (batch_size, latent_size)
        with torch.no_grad():
            latent_hidden = self.latent_to_dec_hidden(latent_z)
            latent_hidden = rearrange(
                latent_hidden, "b (l h) -> l b h ",
                l=self.encoder_decoder_num_layers,
                h=self.encoder_decoder_hidden_size
            )

            X = torch.zeros()

            packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

            output, _ = self.decoder(packed_X, latent_hidden)

            # unpack
            out_X, out_len_X = pad_packed_sequence(output, batch_first=True)

            out_X = self.dec_output_to_vocab(out_X)
            # Log softmax returns large negative numbers for low probas and near-zero for high probas
            # last dim is actual embeddings
            out_X = F.log_softmax(out_X, dim=-1)


def pad_collate(batch):
    (x_batch, y_batch) = zip(*batch)

    len_x = torch.tensor([len(x) for x in x_batch]).long()
    len_y = torch.tensor([len(y) for y in y_batch]).long()
    # TODO MAKE SURE THIS VALUE CORRESPONDS TO THE PAD_IDX
    x_pad = pad_sequence(x_batch, batch_first=True, padding_value=0).to(DEVICE)
    y_pad = pad_sequence(y_batch, batch_first=True, padding_value=0).to(DEVICE)

    return x_pad, y_pad, len_x, len_y


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
    loss = (nll_loss + kl_div * kl_weight) / predictions.size(0)
    return loss, (nll_loss, kl_div, kl_weight)


def train():
    dataset = ABCInMemoryDataset(CONFIG["path_to_abc"], max_len=CONFIG["max_len"], cut_or_filter="cut")

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


def evaluate(path):
    print(f"Loading model {model_path}")
    state = torch.load(model_path)
    CONFIG = state["CONFIG"]
    model = state["model"]
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    eos_tok = "<EOS>"

    TRIES = 500
    print(model)
    print(f"Evaluating model for {TRIES} tries")
    base_folder = Path(model_path).parent
    tunes_out_folder = base_folder.joinpath("output_tunes")
    Path.mkdir(tunes_out_folder, exist_ok=True)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
    ("path_to_abc", "../data_processed/abc_parsed_cleanup2.abc"),
    ("batch_size", 64),
    ("lr", 0.001),
    ("embedding_size", 256),
    ("latent_vector_size", 256),
    ("encoder_decoder_hidden_size", 512),
    ("encoder_decoder_num_layers", 2),
    ("dropout_prob", 0.4),
    ("epochs", 10),
    ("cut_or_filter", "cut"),
    ("max_len", 700)
])

if __name__ == "__main__":
    setup_matplotlib_style()
    mode = "train"
    # mode = "eval"
    model_path = "experiment_20220720-162527/RNNSimpleGenerator.pth"
    if mode == "train":
        train()
    elif mode == "eval":
        evaluate(model_path)
