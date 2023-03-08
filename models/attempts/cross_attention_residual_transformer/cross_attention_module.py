import torch.nn as nn
import torch
from torch.nn.functional import relu


class CrossAttention(nn.Module):
    """Module that implements the Cross attention"""

    def __init__(self, encoder, src_embed, enc_extra_embed, generator) -> None:
        super(CrossAttention, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.enc_extra_embed = enc_extra_embed
        self.generator = generator
        self.out = nn.Embedding(self.encoder.size*2, 512)

    def forward(self, src, obd_spd, src_mask, obd_enc_mask):
        """perform the cross_attention and return two input"""
        embed_spd = self.enc_extra_embed(obd_spd.permute(0, 2, 1))
        code, mix = self.encode(self.src_embed(src), embed_spd, src_mask, obd_enc_mask)
        return self.out(code+mix)

    def encode(self, src, embed_spd, src_mask, obd_enc_mask):
        return self.encoder(self.src_embed(src), embed_spd, src_mask, obd_enc_mask)
