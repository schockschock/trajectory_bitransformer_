import torch.nn as nn
import torch
from torch.nn.functional import relu


class CrossAttention(nn.Module):
    """Module that implements the Cross attention"""

    def __init__(self, encoder) -> None:
        super(CrossAttention, self).__init__()
        self.encoder = encoder
        self.out = nn.Embedding(512, 512)

    def forward(self, src_trj, src_vsn, src_mask, obd_enc_mask):
        """perform the cross_attention and return two input"""
        code, mix = self.encoder(src_trj, src_vsn, src_mask, obd_enc_mask)
        return code+mix
