import torch.nn as nn
from .functional import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    """Core encoder composed of  n layer of cross-attention and a normalisation layer"""

    def __init__(self, layer, n) -> None:
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.input_size)

    def forward(self, x, embed_spd, src_mask, obd_enc_mask):
        """Pass the input into the layers"""
        t = 0
        m = embed_spd
        for layer in self.layers:
            x, m = layer(x, m, src_mask, obd_enc_mask, t)
            t += 1
        return self.norm(x), self.norm(m)
