import torch.nn as nn
import torch.nn.functional as F
import torch
from .encoder import TransformerBlock
from .Attention import MultiHeadAttention
from .components import PositionalEncoding


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


#Pour l'embedding
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out
#######################

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8) -> None:
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(
            embed_dim, expansion_factor, n_heads)

    def forward(self, key, x, value, mask):

        attention = self.attention(x, x, x, mask=mask)  # 32x12x512
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_size, embed_dim, seq_len, mask=None,num_layers=2, expansion_factor=4, n_heads=8) -> None:
        super(TransformerDecoder, self).__init__()
        self.fc_embedding = nn.Linear(2,embed_dim)
        self.relu = nn.ReLU(inplace=False)
        #self.embedding = Embedding(embed_dim, target_size)
        self.dropout = nn.Dropout(0.1)
        self.positional_encoding = PositionalEncoding(embed_dim, self.dropout)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, 2)
        

    def forward(self, x, enc_out, mask):

        x = self.fc_embedding(x)
        #x = self.relu(x)
        #x = self.embedding(x)
        
        x = self.positional_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:

            x = layer(enc_out, x, enc_out, mask)
        
        out = self.fc_out(x)
        
        return out