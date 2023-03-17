import torch
import torch.nn as nn
from .cross_attention_residual_transformer.encoder import Encoder
from .cross_attention_residual_transformer.encoder_layer import EncoderLayer
from .cross_attention_residual_transformer.cross_attention_module import CrossAttention
from .cross_attention_residual_transformer.layers import MultiHeadAttention,  PositionalEncoding, PointerwiseFeedforward
import copy
import math

#from .scratch_transformer.decoder import TransformerDecoder


class crossAttention(nn.Module):
    def __init__(self, N=1,
                 d_model=256, d_ff=2048, h=8, dropout=0.1) -> None:
        super(crossAttention, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = CrossAttention(
            Encoder(EncoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N))
        self.enc_embed_dim = d_ff

    def forward(self, src_trj, src_vsn, src_mask, obd_enc_mask):
        return self.encoder(src_trj, src_vsn, src_mask, obd_enc_mask)


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LinearEmbedding_sp(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding_sp, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)
