from sentence_encoder import SentenceEncoder
import torch
import torch.nn as nn
from typing import Optional


class RNNBaseline(SentenceEncoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: Optional[bool] = True,
                 num_layers: Optional[int] = 1):
        super(RNNBaseline, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            batch_first=True,
                            num_layers=num_layers)

        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = num_layers

    def get_output_dim(self):
        return self.lstm.hidden_size * self.num_directions

    def _get_lstm_features(self, x: torch.Tensor, x_lengths: torch.IntTensor) -> torch.Tensor:
        # Ignore padding in LSTM
        x = nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Feed into LSTM
        x, _ = self.lstm(x)
        # Undo packaging
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x

    def forward(self, x: torch.Tensor, x_lengths: Optional[torch.IntTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Feed into LSTM
        x = self._get_lstm_features(x, x_lengths)

        # Return the mean along second dim (taking into account padding)
        return torch.sum(x, dim=1) / x_lengths.unsqueeze(dim=1)
