import torch.nn as nn
import torch.nn.functional as F
import torch
from encoder import TransformerBlock
from Attention import MultiHeadAttention
from components import Embedding, PositionalEncoding


def make_trg_mask(trg):
    """
    Args:
        trg: target sequence
    Returns:
        trg_mask: target mask
    """
    batch_size, trg_len = trg.shape
    # returns the lower triangular part of matrix filled with ones
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, 1, trg_len, trg_len
    )
    return trg_mask


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8) -> None:
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(
            embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        attention = self.attention(x, x, x, mask=mask)  # 32x12x512
        value = self.dropout(self.norm(attention+x))
        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_size, embed_dim, num_layers=2) -> None:
        super(TransformerDecoder, self).__init__()

        self.embedding = Embedding(embed_dim, target_size)
        self.positional_encoding = PositionalEncoding(embed_dim, 0.1)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, target_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = self.fc_out(x)
