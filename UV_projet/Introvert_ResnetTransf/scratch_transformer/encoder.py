import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import math
from .Attention import MultiHeadAttention
from .components import Embedding, PositionalEncoding


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)  # 32x8x512
#         print("#####")
#         print(attention_out.size())
#         print(value.size())
#         print("#####")
        attention_residual_out = attention_out + value  # 32x8x512
        norm1_out = self.dropout1(self.norm1(
            attention_residual_out))  # 32x8x512

        # 32x8x512 -> #32x8x2048 -> 32x8x512
        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x8x512
        norm2_out = self.dropout2(self.norm2(
            feed_fwd_residual_out))  # 32x8x512

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """

    def __init__(self, seq_len, input_size, embed_dim, num_layer=2, expansion_factor=4, n_heads=8) -> None:
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(embed_dim, input_size)
        self.positional_encoding = PositionalEncoding(embed_dim, 0.1)

        self.layers = nn.ModuleList([TransformerBlock(
            embed_dim, expansion_factor, n_heads) for _ in range(num_layer)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoding(embed_out)
        for layer in self.layers:
            out = layer(out)
        return out  # 32x8x512
