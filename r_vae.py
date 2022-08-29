import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from models.RecurrentEncoder import RecurrentEncoder
from models.VAEMuLogVar import VAEMuLogVar
from losses import padded_kl_nll_loss
from utils import setup_matplotlib_style, plot_training_history_from_pickle, save_training_attempt, init_history, \
    pad_collate

#### !!!!!
## TODO
#   THIS ONE USES THE CONCAT OF Z TO X
class RVAE(torch.nn.Module):
    def __init__(
            self,
            vocab,
            embedding_dim,
            enc_dec_hidden,
            enc_dec_nlayers,
            latent_dim,
            dropout_prob,
            bidirectional_encoder=False
    ):
        super(RVAE, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.enc_dec_hidden = enc_dec_hidden
        self.enc_dec_nlayers = enc_dec_nlayers
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=self.embedding_dim
        )

        self.encoder = RecurrentEncoder(
            input_size=self.embedding_dim,
            hidden_dim=self.enc_dec_hidden,
            n_layers=self.enc_dec_nlayers,
            rnn_type="GRU",
            bidirectional=bidirectional_encoder
        )

        self.bidi_multiplier = 2 if bidirectional_encoder else 1

        self.mulogvar = VAEMuLogVar(
            in_features=self.bidi_multiplier * self.enc_dec_hidden * self.enc_dec_nlayers,
            latent_dim=self.latent_dim
        )

        self.latent_to_dec_hidden = nn.Sequential(
            nn.Linear(self.latent_dim, self.enc_dec_hidden * self.enc_dec_nlayers * self.bidi_multiplier),
            nn.Tanh()
        )

        self.decoder = torch.nn.GRU(
            input_size=self.embedding_dim + self.latent_dim,
            hidden_size=self.bidi_multiplier * self.enc_dec_hidden,
            num_layers=self.enc_dec_nlayers,
            batch_first=True
        )

        self.emb_dropout = torch.nn.Dropout(p=self.dropout_prob)

        self.dec_output_to_vocab = torch.nn.Linear(self.bidi_multiplier * self.enc_dec_hidden, len(vocab))


    def forward(self, X, len_X):
        batch_size, padded_sequence_len = X.size()
        X = self.embedding(X) # X = (batch size, max_len, embedding_size)
        _, hidden = self.encoder(X, len_X, DEVICE)  # hidden = ( bidi * num_layers, batch_size, hidden_size)

        hidden = rearrange(
            hidden, "(bidi l) b h -> b (l bidi h)",
            bidi=self.bidi_multiplier
        )

        z, mu, logvar = self.mulogvar(hidden, DEVICE)

        latent_hidden = self.latent_to_dec_hidden(z) #
        latent_hidden = rearrange(
            latent_hidden, "b (l bidi h) -> l b (bidi h)",
            b=batch_size, h=self.enc_dec_hidden, bidi=self.bidi_multiplier
        )
        decoder_z = repeat(z, "b z -> b repeat z", repeat=X.size(1))
        decoder_input = torch.cat([X, decoder_z], dim=-1)

        packed_X = pack_padded_sequence(decoder_input, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)

        output, _ = self.decoder(packed_X, latent_hidden)

        # unpack
        out_X, out_len_X = pad_packed_sequence(output, batch_first=True)

        out_X = self.dec_output_to_vocab(out_X)
        # Log softmax returns large negative numbers for low probas and near-zero for high probas
        # last dim is actual embeddings
        out_X = F.log_softmax(out_X, dim=-1)

        return out_X, out_len_X, mu, logvar, z

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
                # X = rearrange(X, "x -> 1 x")
                X = self.embedding(X)
                X = torch.cat([X, latent_z], dim=-1)
                X = rearrange(X, "e x -> 1 e x")
                # X = torch.cat([X, latent_z], dim=-1)

                output, latent_hidden = self.decoder(X, latent_hidden)

                out_X = self.dec_output_to_vocab(output)
                # Log softmax returns large negative numbers for low probas and near-zero for high probas
                # last dim is actual embeddings
                probas = F.softmax(out_X, dim=-1).cpu().squeeze().numpy()

                chosen_index = np.random.choice(len(out_X.squeeze()), p=probas)

                # chosen_index = np.argmax(probas)

                X = torch.tensor([chosen_index], device=DEVICE).long()
                # X = torch.argmax(out_X, dim=-1)

                generated_tune.extend(self.vocab.lookup_tokens([chosen_index]))
                i += 1
                if i >= 1000:
                    return "".join(generated_tune)

        return "".join(generated_tune).replace("<EOS>", "")


def train():
    dataset = ABCInMemoryDataset(
        CONFIG["path_to_abc"],
        min_len=CONFIG["min_len"],
        max_len=CONFIG["max_len"],
        cut_or_filter=CONFIG["cut_or_filter"]
    )

    collate = partial(pad_collate, device=DEVICE, pad_idx=dataset.PAD_IDX)

    train_data_loader, valid_data_loader, test_data_loader = split_train_valid_test_dataloaders(
        dataset, train_percent=0.8, valid_percent=0.1,
        batch_size=CONFIG["batch_size"],
        collate_fn=collate)

    loss_fn = partial(padded_kl_nll_loss, device=DEVICE)

    model = RVAE(
        vocab=dataset.vocabulary,
        embedding_dim=CONFIG["embedding_size"],
        enc_dec_hidden=CONFIG["encoder_decoder_hidden_size"],
        enc_dec_nlayers=CONFIG["encoder_decoder_num_layers"],
        latent_dim=CONFIG["latent_vector_size"],
        dropout_prob=CONFIG["dropout_prob"],
        bidirectional_encoder=CONFIG["bidirectional_encoder"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08
    )

    history = init_history()
    model.train()
    optimizer_step = 0

    # kl_history = train/val, epoch, batch, (nll_loss, kl_div, kl_weight)
    kl_history = torch.tensor(np.zeros((2, CONFIG["epochs"], len(train_data_loader), 3)), device=DEVICE)
    for epoch in range(CONFIG["epochs"]):
        epoch_start_time = time.perf_counter()
        minibatch_losses = torch.tensor(np.zeros(len(train_data_loader)), device=DEVICE)

        for i, (batch_X, batch_y, len_X, len_y) in enumerate(tqdm.tqdm(train_data_loader)):

            model.zero_grad(set_to_none=True)
            predictions, len_predictions, mean, log_variance, z = model(batch_X, len_X)
            targets = batch_y.long()
            loss, (nll_loss, kl_div, kl_weight) = loss_fn(
                predictions, len_predictions, targets, len_y, mean, log_variance, optimizer_step
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
                val_loss, (nll_loss, kl_div, kl_weight) = loss_fn(
                    predictions, len_predictions, targets, len_y, mean, log_variance, optimizer_step
                )
                val_minibatch_losses[i] = val_loss

        history["val"]["loss"].append(torch.nanmean(val_minibatch_losses).item())
        epoch_time = time.perf_counter() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        tqdm.tqdm.write(
            f"Epoch {epoch + 1}: [{epoch_time:.2f}s] Train loss: {history['train']['loss'][epoch]} | Val loss: {history['val']['loss'][epoch]}")

        scheduler.step(val_loss)

    save_training_attempt(model, CONFIG, history, np_extra={"kl_history": kl_history.cpu().numpy()})


def evaluate(model_path):
    print(f"Loading model {model_path}")
    state = torch.load(model_path)
    model = state["model"]
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    bos_idx = model.vocab.lookup_indices(["<BOS>"])[0]
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
        #generated_tune = model.generate(latent_z, bos_idx, unk_idx)
        generated_tune = model.generate_tok_by_tok(latent_z, bos_idx, eos_idx)
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

def interpolate(model, tune_1 = None, tune_2 = None):
    dataset = ABCInMemoryDataset(
        CONFIG["path_to_abc"],
        min_len=CONFIG["min_len"],
        max_len=CONFIG["max_len"],
        cut_or_filter=CONFIG["cut_or_filter"]
    )

    print(f"Loading model {model_path}")
    state = torch.load(model_path)
    model = state["model"]
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tune_1 = """
M:6/8
L:1/8
K:B
|CBBC|ABBA|
"""

    tune_2 = """
M:4/4
L:1/8
K:G
|CGDA|AGAB|
    """
    tune_1 = torch.tensor(dataset.text_to_int_encoder(list(tune_1)), device=DEVICE, requires_grad=False)
    tune_2 = torch.tensor(dataset.text_to_int_encoder(list(tune_2)), device=DEVICE, requires_grad=False)

    with torch.no_grad():
        #make into batch of 1
        tune_1 = rearrange(tune_1, "t -> 1 t")
        tune_2 = rearrange(tune_2, "t -> 1 t")
        len_tune_1 = [len(t) for t in tune_1]
        len_tune_2 = [len(t) for t in tune_2]
        tune_1 = model.embedding(tune_1)

        _, hidden = model.encoder(tune_1, len_tune_1, DEVICE)

        hidden = rearrange(
            hidden, "(bidi l) b h -> b (l bidi h)",
            bidi=model.bidi_multiplier
        )

        latent_1, _, _ = model.mulogvar(hidden, DEVICE)

        tune_2 = model.embedding(tune_2)

        _, hidden = model.encoder(tune_2, len_tune_2, DEVICE)

        hidden = rearrange(
            hidden, "(bidi l) b h -> b (l bidi h)",
            bidi=model.bidi_multiplier
        )

        bos_idx = model.vocab.lookup_indices(["<BOS>"])[0]
        eos_idx = model.vocab.lookup_indices(["<EOS>"])[0]

        latent_2, _, _ = model.mulogvar(hidden, DEVICE)
        n_intermediate = 5
        interpolated = np.linspace(latent_1.cpu().numpy(), latent_2.cpu().numpy(), n_intermediate + 2)
        for z in interpolated:
            generated_tune = model.generate_tok_by_tok(torch.from_numpy(z).to(DEVICE), bos_idx, eos_idx)
            print(generated_tune)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
    ("path_to_abc", "../data_processed/abc_parsed_cleanup5.abc"),
    ("batch_size", 40),
    ("lr", 0.001),
    ("embedding_size", 32),
    ("latent_vector_size", 256),
    ("encoder_decoder_hidden_size", 512),
    ("encoder_decoder_num_layers", 3),
    ("dropout_prob", 0.5),
    ("epochs", 20),
    ("cut_or_filter", "cut"),
    ("min_len", 30),
    ("max_len", 400),
    ("bidirectional_encoder", False)
])

if __name__ == "__main__":
    setup_matplotlib_style()
    mode = "train"
    # mode = "eval"
    model_path = "experiment_RVAE_20220808-171407/RVAE.pth"
    if mode == "train":
        train()
    elif mode == "eval":
        evaluate(model_path)
        # interpolate(model_path)

