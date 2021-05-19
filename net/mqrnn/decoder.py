import numpy as np
from net.mlp import MLP
import torch
from torch import nn


class _LocalMLP(nn.Module):
    def __init__(self, global_context_size, local_context_size, n_future_covariates, n_quantiles, n_local_layers):
        super().__init__()

        self._global_context_size = global_context_size
        self._local_context_size = local_context_size
        self._n_quantiles = n_quantiles

        self._in_size = global_context_size + local_context_size + n_future_covariates
        out_size = n_quantiles
        self.mlp = MLP(self._in_size, np.linspace(self._in_size, out_size, n_local_layers, dtype=int))

    def forward(self, global_context, local_context, future_covariates):
        local_input = torch.cat([global_context, local_context, future_covariates], dim=1)
        return self.mlp(local_input)



class _GlobalMLP(nn.Module):
    def __init__(self, state_size, n_future_covariates, n_horizons, local_context_size, global_context_size,
                 n_global_layers):
        super().__init__()

        self._global_context_size = global_context_size
        self._local_context_size = local_context_size
        self._n_horizons = n_horizons

        self._in_size = state_size + n_future_covariates * n_horizons
        out_size = n_horizons * local_context_size + global_context_size
        self.mlp = MLP(self._in_size, np.linspace(self._in_size, out_size, n_global_layers, dtype=int))

    def _merge_inputs_for_global_mlp(self, state, future_covariates):
        future_covariates = [cov[:, :self._n_horizons] for cov in future_covariates]
        return torch.cat([state, *future_covariates], dim=1)

    def _split_mlp_outputs_to_contexts(self, mlp_output):
        batch_size = mlp_output.size(0)
        global_context = mlp_output[:, :self._global_context_size]
        local_contexts = mlp_output[:, self._global_context_size:] \
            .reshape(batch_size, -1, self._local_context_size)

        local_contexts = torch.transpose(local_contexts, 0, 1)

        return global_context, local_contexts

    def forward(self, state: torch.Tensor, future_covariates: 'Iterable[torch.Tensor]'):
        global_input = self._merge_inputs_for_global_mlp(state, future_covariates)
        mlp_output = self.mlp(global_input)

        # Expected output: ((batch, global_context) , (n_horizons, batch, local_context))
        return self._split_mlp_outputs_to_contexts(mlp_output)


class DecoderNet(nn.Module):
    def __init__(self, state_size, n_future_covariates, n_quantiles, n_horizons,
                 local_context_size=16, global_context_size=32, n_global_layers=3, n_local_layers=2):
        super().__init__()

        self._n_horizons = n_horizons
        self._n_quantiles = n_quantiles

        self._global_mlp = _GlobalMLP(state_size, n_future_covariates, n_horizons, local_context_size,
                                     global_context_size,
                                     n_global_layers)
        self._local_mlp = _LocalMLP(global_context_size, local_context_size, n_future_covariates, n_quantiles,
                                   n_local_layers)

    @staticmethod
    def _standardize_future_covariates(future_covariates):

        if len(future_covariates) == 1:
            if future_covariates[0].ndim == 2:
                return torch.as_tensor(future_covariates)
            else:
                assert future_covariates[0].ndim == 3, print(future_covariates[0])
                return future_covariates[0]

        reshaped_covs = []
        for cov in future_covariates:
            if cov.ndim == 1:
                reshaped_covs.append(cov)
            else:
                for i in range(cov.ndim):
                    reshaped_covs.append(cov[:, i])

        return torch.stack(reshaped_covs)

    def forward(self, state: torch.Tensor, *future_covariates: torch.Tensor):

        future_covariates = self._standardize_future_covariates(future_covariates)
        c_a, c_ts = self._global_mlp(state, future_covariates)

        batch_size = state.size(0)
        rets = torch.empty(self._n_horizons, batch_size, self._n_quantiles, device=state.device)
        for horizon in range(self._n_horizons):
            rets[horizon] = self._local_mlp(c_a, c_ts[horizon], future_covariates[:, :, horizon].T)

        # Expected output: (n_horizons, batch , n_quantiles)
        return rets

if __name__ == '__main__':
    dec = DecoderNet(state_size=31,
                     n_future_covariates=3,
                     n_quantiles=5,
                     n_horizons=7,
                     local_context_size=11,
                     global_context_size=23)

    encoder_state = torch.rand(64, 31)
    future_covariates = torch.rand(3, 64, 7)

    ret = dec(encoder_state, future_covariates)
    print(ret.shape)
