import torch
from torch import nn
from torch.nn import functional as F


class QuantileLoss(nn.Module):
    __constants__ = ['q']
    q: torch.Tensor

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('q', torch.tensor(quantiles).reshape(-1))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # input size: (batch x quantiles)
        # input size: (batch)

        target = target.reshape(-1, 1)
        difference = input - target
        return torch.mean(torch.sum(self.q * F.relu(difference) + (1. - self.q) * F.relu(-difference), axis=1))
