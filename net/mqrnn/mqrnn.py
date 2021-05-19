from contextlib import contextmanager
import torch

from typing import Iterable
from torch import nn
from .decoder import DecoderNet
from .lstm_encoder import EncoderNet


class MQRNN(nn.Module):
    def __init__(self, n_covariates: int, quantiles: Iterable[float], n_horizons: int, hist_len: int,
                 encoder_state_size=32, local_context_size=8, global_context_size=16,
                 n_decoder_global_layers=5, n_decoder_local_layers=3, forking_sequences: bool = True):

        super(MQRNN, self).__init__()

        quantiles = tuple(quantiles)

        self.hist_len = hist_len
        self.n_horizons = n_horizons
        self.forking_sequences = forking_sequences

        # To avoid unnecessary transpositions, we set this value to `True`
        self._return_final_result = True

        self.encoder = EncoderNet(1 + n_covariates, encoder_state_size)
        self.decoder = DecoderNet(encoder_state_size, n_covariates, len(quantiles), n_horizons,
                                  local_context_size, global_context_size, n_decoder_global_layers,
                                  n_decoder_local_layers)

    @contextmanager
    def non_final_context(self):
        try:
            self._return_final_result = False
            yield
        finally:
            self._return_final_result = True

    def decode_for_time_t(self, states: torch.Tensor, time_first_covs: torch.Tensor, t: int):
        # shapes:
        #   states:          (n_hidden_states, batch, state_size)
        #   time_first_covs: (time, batch, # covariates)

        fut_covs = time_first_covs[t: t + self.n_horizons].transpose(0, 2)

        # Expeceted output: (n_horizons, batch , n_quantiles)
        return self.decoder(states[t], fut_covs)

    @staticmethod
    def _merge_data_and_covs(data, time_first_covs):
        # expected shapes:
        #   data: (batch, hist)
        #   covs: (hist + future, batch, # covariates)

        hist_len = data.size(1)

        merged = torch.cat([data.transpose(0, 1).unsqueeze(-1), time_first_covs[:hist_len]], dim=2)
        return merged

    @property
    def return_history(self):
        return self.training and self.forking_sequences

    def forward(self, data: torch.Tensor, covs: torch.Tensor):
        # expected shapes:
        #   data: (batch, hist, [1])
        #   covs: (batch, hist + future, [# covariates])

        data = data.to(self.device)
        covs = covs.to(self.device)

        # Fix dimensions
        if data.ndim == 3:
            data.squeeze_(2)
            assert data.ndim == 2

        if covs.ndim == 2:
            covs.unsqueeze_(-1)

        # Concatenate all the data to a single block
        time_first_covs = covs.transpose(0, 1)

        inp = self._merge_data_and_covs(data, time_first_covs)
        states = self.encoder(inp)

        if not self.return_history:
            # Expeceted output: (n_horizons, batch, n_quantiles)
            result = self.decode_for_time_t(states, time_first_covs, self.hist_len - 1)

            if self._return_final_result:
                return result.transpose(0, 1)

            return result

        results = []
        for t in range(self.hist_len):
            results.append(self.decode_for_time_t(states, time_first_covs, t))

        # Expeceted output: (hist, n_horizons, batch, n_quantiles)
        return torch.stack(results)


if __name__ == '__main__':

    mqrnn = MQRNN(n_covariates=3,
                  quantiles=[.1, .5, .9],
                  n_horizons=24,
                  hist_len=168)

    all_data = torch.rand(64, 192)
    inp = all_data[:, :168]
    covariates = torch.rand(64, 192, 3)

    # Expeceted output: (hist, n_horizons, batch, n_quantiles)
    results = mqrnn(inp, covariates)
    print(results)