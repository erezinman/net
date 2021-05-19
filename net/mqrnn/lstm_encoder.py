import torch
from torch import nn


class EncoderNet(nn.Module):
    def __init__(self, input_channels: int, hidden_units: int):
        super().__init__()

        self.input_size = input_channels
        self.hidden_units = hidden_units

        self._lstm = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_units)
        self._state = None

    def forward(self, input):
        # input shape: (time, batch, n_time_series)
        output, (h_n, c_n) = self._lstm(input, self._state)
        # output shape: (time, batch, hidden_units)
        return output
