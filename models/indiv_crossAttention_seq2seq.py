import torch
import torch.nn as nn
from .cross_attention_transformer.encoder import Encoder
from .cross_attention_transformer.encoder_layer import EncoderLayer
from .cross_attention_transformer.cross_attention_module import CrossAttention
from .cross_attention_transformer.layers import MultiHeadAttention,  PositionalEncoding, PointerwiseFeedforward
import copy
import math

from .scratch_transformer.decoder import TransformerDecoder


class crossAttention_seq2seq(nn.Module):
    def __init__(self, inp_length, enc_inp_size, dec_inp_size, dec_out_size, N=1,
                 d_model=512, d_ff=2048, h=8, dropout=0.1) -> None:
        super(crossAttention_seq2seq, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = CrossAttention(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size, d_model), c(position)),
            nn.Sequential(LinearEmbedding_sp(2*(inp_length-1), d_model)))
        self.enc_embed_dim = d_ff

        # decoder transformer
        self.decoder = TransformerDecoder(dec_out_size, self.enc_embed_dim)

    def forward(self, src, obd_spd, src_mask, obd_enc_mask):
        code = self.encoder(src, obd_spd, src_mask, obd_enc_mask)


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
