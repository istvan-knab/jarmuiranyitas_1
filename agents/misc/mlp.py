import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, size: list, seed: int):
        super(MultiLayerPerceptron, self).__init__()

        self._seed(seed)

        self.layers = {}
        for x in range(len(size)-1):
            self.layers["linear_{0}".format(x+1)] = nn.Linear(size[x], size[x+1])

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear_1(x)
        for i in range(len(self.layers)-1):
            out = self.relu(out)
            out = self.layers["linear_{0}".format(i+2)](out)

        return out

    @staticmethod
    def _seed(seed):
        torch.manual_seed(seed)
