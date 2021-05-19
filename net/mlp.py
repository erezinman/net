import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_sizes, activation=nn.ReLU):
        super().__init__()

        self.activation = activation()
        self.modules_list = nn.ModuleList()

        for output_size in output_sizes:
            self.modules_list.append(nn.Linear(input_size, output_size))
            input_size = output_size

    def forward(self, input):
        first = True
        for layer in self.modules_list:
            if not first:
                input = self.activation(input)
                first = False
            input = layer(input)
        return input
