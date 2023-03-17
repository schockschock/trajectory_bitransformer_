import torch.nn as nn
from .layer_norm import LayerNorm


class SubLayerConnection(nn.Module):
    """Layer composed of a residual connectino followed by a normalisation"""

    def __init__(self, input_size, dropout) -> None:
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

    def src_attention(self, x, sublayer, sublayer1, sublayer2):
        x1 = sublayer(self.norm(x))
        x2 = sublayer1(self.norm(x))
        x3 = sublayer2(self.norm(x))
        return x + self.dropout(x1)+self.dropout(x2)+self.dropout(x3)

    def forward(self, x, sublayer, st=1, sublayer1=None, sublayer2=None):
        """apply residual connection to sublayer with same size"""
        if st == 0:
            return x + self.dropout(sublayer(x))
        if sublayer1 == None:
            return self.self_attention(x, sublayer)
        else:
            return self.src_attention(x, sublayer, sublayer1, sublayer2)
