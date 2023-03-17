import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """construct a layer that performs normalisation"""

    def __init__(self, features_shape, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features_shape))
        self.b_2 = nn.Parameter(torch.ones(features_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        sdt = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x-mean) / (self.std+self.eps) + self.b_2
