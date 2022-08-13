import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentEncoder(nn.Module):

    def __init__(
            self,
            input_size: int,
            n_layers: int,
            hidden_dim: int,
            bidirectional: bool,
            rnn_type: str = "GRU",
            **kwargs
    ) -> None:
        super().__init__()

        rnn = {
            "GRU": torch.nn.GRU,
            "LSTM": torch.nn.LSTM
        }
        self.bidirectional = bidirectional
        self.encoder = rnn[rnn_type](
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            **kwargs
        )

    def forward(self, X, len_X, device, use_packed=False):
        # X = (batch size, max_len, embedding_size)
        # len_X = [len(x) for x in X]
        if use_packed:
            X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(device)
            # output = (batch_size, sequence_length, D * hidden_dim)
            # hidden = (D * num_layers, batch_size, hidden_dim)
        output, hidden = self.encoder(X)
        # if self.bidirectional:
        #     hidden = rearrange(hidden, "(2 n) b h -> n b (2 h)")
        return output, hidden
