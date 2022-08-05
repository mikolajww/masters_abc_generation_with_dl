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
            rnn_type: str = "GRU",
            **kwargs
    ) -> None:
        super().__init__()

        rnn = {
            "GRU": torch.nn.GRU,
            "LSTM": torch.nn.LSTM
        }
        self.encoder = rnn[rnn_type](
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            **kwargs
        )


    def forward(self, X, len_X, device):
        # X = (batch size, max_len, embedding_size)
        # len_X = [len(x) for x in X]
        packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(device)

        # output = (batch size, sequence length, hidden_dim)
        output, hidden = self.encoder(packed_X)
        hidden = rearrange(hidden, "l b h -> b (l h)")
        return output, hidden
    