import pickle
import pprint
import time
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tqdm
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from losses import padded_crossentropy_loss
from utils import setup_matplotlib_style, is_valid_abc, plot_training_history_from_pickle, save_training_attempt, \
    init_history, pad_collate
from functools import partial


class RNNSimpleGenerator(nn.Module):
    def __init__(
            self,
            vocab,
            input_features,
            hidden_size,
            n_layers,
            dropout_prob,
            one_hot=True,
            embed_dim=32
    ):
        super(RNNSimpleGenerator, self).__init__()

        self.vocab = vocab
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_prob if hidden_size > 1 else 0,
            bidirectional=False
        )
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.one_hot = one_hot

        if hidden_size == 1:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(hidden_size, len(self.vocab))
        self.embed = nn.Embedding(len(self.vocab), embed_dim)

    def forward(self, X, len_X, initial_hidden=None):
        if self.one_hot:
            X = F.one_hot(X.long(), num_classes=len(self.vocab)).float()
        else:
            X = self.embed(X)
        X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)
        X, h = self.lstm(X, initial_hidden)
        X, len_X = pad_packed_sequence(X, batch_first=True)
        X = self.dropout(X)
        X = self.fc(X)
        return X, len_X, h

    def init_hidden(self, batch_size, device):
        h = torch.zeros(size=(self.n_layers, batch_size, self.hidden_size)).to(device)
        c = torch.zeros(size=(self.n_layers, batch_size, self.hidden_size)).to(device)
        return h, c


def train():
    dataset = ABCInMemoryDataset(CONFIG["path_to_abc"], max_len=CONFIG["max_len"], cut_or_filter="cut")

    collate = partial(pad_collate, device=DEVICE, pad_idx=dataset.PAD_IDX)

    train_data_loader, valid_data_loader, test_data_loader = split_train_valid_test_dataloaders(
        dataset, train_percent=0.8, valid_percent=0.1, batch_size=CONFIG["batch_size"], collate_fn=collate)

    model = RNNSimpleGenerator(
        input_features=len(dataset.vocabulary),
        vocab=dataset.vocabulary,
        hidden_size=CONFIG["hidden_size"],
        n_layers=CONFIG["n_recurrent_layers"],
        dropout_prob=CONFIG["dropout_prob"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08
    )

    model.train()

    loss_fn = partial(padded_crossentropy_loss, device=DEVICE)
    history = init_history()

    for epoch in range(CONFIG["epochs"]):
        epoch_start_time = time.perf_counter()

        minibatch_losses = torch.tensor(np.zeros(len(train_data_loader)), device=DEVICE)
        for i, (batch_X, batch_y, len_X, len_y) in enumerate(tqdm.tqdm(train_data_loader)):
            # this is equivalent to optimizer.zero_grad()
            # reset gradients every minibatch!
            # set_to_none supposedly is more efficient as per
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)

            # pass data through a model
            output_packed, len_packed, _ = model(batch_X, len_X)
            targets = batch_y.long()

            # calculate loss and backpropagate gradients
            loss = loss_fn(output_packed, len_packed, targets, len_y)
            minibatch_losses[i] = loss.data
            loss.backward()

            # Prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if i % 50 == 0:
                tqdm.tqdm.write(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {loss.item()}")

        history["train"]["loss"].append(torch.nanmean(minibatch_losses).item())

        # Evaluate on validation dataset without gradients
        with torch.no_grad():
            val_minibatch_losses = torch.tensor(np.zeros(len(valid_data_loader)), device=DEVICE)
            for i, (batch_X, batch_y, len_X, len_y) in enumerate(valid_data_loader):
                output_packed, len_packed, _ = model(batch_X, len_X)

                targets = batch_y.long()
                val_loss = loss_fn(output_packed, len_packed, targets, len_y)
                val_minibatch_losses[i] = val_loss.data

        history["val"]["loss"].append(torch.nanmean(val_minibatch_losses).item())

        epoch_time = time.perf_counter() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        tqdm.tqdm.write(
            f"Epoch {epoch + 1}: [{epoch_time:.2f}s] Train loss: {history['train']['loss'][epoch]} | Val loss: {history['val']['loss'][epoch]}")

        scheduler.step(val_loss)

    save_training_attempt(model, CONFIG, history)


def try_generate(model, eos):
    start_character = "<BOS>"
    attempt = 0
    tries = []
    char2int = lambda t: model.vocab.lookup_indices(t)
    int2char = lambda t: model.vocab.lookup_tokens(t)
    with torch.no_grad():
        while True:
            generated_tokens = [start_character]
            attempt += 1

            h = model.init_hidden(batch_size=1, device=DEVICE)
            c = model.init_hidden(batch_size=1, device=DEVICE)
            generated_tok = None
            while generated_tok != eos:
                x = torch.tensor(char2int([generated_tokens[-1]])).expand(1, 1)  # 1 element batch
                if x.ndimension() == 1:
                    len_x = torch.tensor([len(x)]).long()
                else:
                    len_x = [len(t) for t in x]
                    len_x = torch.tensor(len_x).long()
                y_pred, len_y, (h, c) = model(x, len_x, (h, c))
                # unpad
                for i in range(y_pred.size(0)):
                    y_pred[i] = y_pred[i][:len_y[i]]
                # y_pred = (Batch, time step, character (1-hot))
                p = F.softmax(y_pred, dim=2).cpu().numpy().squeeze()

                generated_tok = np.random.choice(y_pred.size(-1), p=p)
                generated_tok = int2char([generated_tok])[0]
                generated_tokens.append(generated_tok)
            generated_tune = "".join(generated_tokens).replace(start_character, "").replace(eos, "")
            if is_valid_abc(generated_tune):
                break
            else:
                tries.append(generated_tune)
        return generated_tune, tries, attempt


def evaluate(model_path):
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

    eval_start = time.perf_counter()
    eval_times = []
    n_of_attempts_list = []
    for i in tqdm.trange(TRIES):
        eval_attempt_start = time.perf_counter()
        generated_tune, tries, n_of_attempts = try_generate(model, eos_tok)
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

    plot_training_history_from_pickle(base_folder, "history.pkl")

    eval_summary_str = f"\nTook {eval_end - eval_start:.2f}s to evaluate. [min = {eval_times.min()}, avg = {eval_times.mean()}, max = {eval_times.max()}]"
    with open(f"{base_folder}/evaluation_summary.txt", "w") as f:
        f.writelines(pprint.pformat(CONFIG))
        f.write("\n")
        f.writelines(str(model))
        f.writelines(model.n)
        f.writelines(eval_summary_str)

    print(eval_summary_str)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
    ("path_to_abc", "../data_processed/abc_parsed_cleanup2.abc"),
    ("batch_size", 64),
    ("lr", 0.002),
    ("hidden_size", 512),
    ("n_recurrent_layers", 3),
    ("dropout_prob", 0.5),
    ("epochs", 20),
    ("cut_or_filter", "cut"),
    ("max_len", 700)
])

if __name__ == "__main__":
    setup_matplotlib_style()
    mode = "train"
    # mode = "eval"
    model_path = "experiment_20220721-150209/RNNSimpleGenerator.pth"
    if mode == "train":
        train()
    elif mode == "eval":
        evaluate(model_path)
    else:
        hist = pickle.load(open("experiment_20220715-161305_no_L_metatag/history.pkl", "rb"))

        print(hist)
