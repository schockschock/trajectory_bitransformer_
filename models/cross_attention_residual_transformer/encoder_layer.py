import torch.nn as nn
import torch

from .functional import clones
from .sublayer_connection import SubLayerConnection


class EncoderLayer(nn.Module):
    """Encoder layer compose of self-attention and feed forward"""

    def __init__(self, input_size, src_attn, cross_attn, feed_forward, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.src_attn = src_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(input_size, dropout), 4)
        self.input_size = input_size

    def forward(self, x, embed, src_mask, obd_enc_mask, st):
        """ Follow connections of multidimensional transformer """
        m = embed
        x = x
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, obd_enc_mask))
        m = self.sublayer[2](m, lambda m: self.cross_attn(m, x, x, src_mask))
        # First is the output of top cross-attention (on x,m,m), second is the output of second cross-attention
        return self.sublayer[0](x, self.feed_forward), self.sublayer[3](m, self.feed_forward)
