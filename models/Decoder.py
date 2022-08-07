import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence


class Decoder(nn.Module):

    def __init__(
            self,
            input_size: int,
            n_layers: int,
            hidden_dim: int,
            output_size: int,
            rnn_type: str = "GRU",

            **kwargs
    ) -> None:
        super().__init__()

        rnn = {
            "GRU": torch.nn.GRU,
            "LSTM": torch.nn.LSTM
        }
        self.decoder = rnn[rnn_type](
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            **kwargs
        )
        self.out_to_vocab_prob = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, X, len_X, device):

        output, _ = self.decoder(packed_X, latent_hidden)
        out_X, out_len_X = pad_packed_sequence(output, batch_first=True)

        out_X = self.dec_output_to_vocab(out_X)

