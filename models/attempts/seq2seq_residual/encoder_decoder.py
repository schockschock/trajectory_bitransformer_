# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn
import torch

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder_trj, encoder_vsn, decoder, src_embed, tgt_embed, generator, code_size):
        super(EncoderDecoder, self).__init__()
        self.encoder_trj = encoder_trj
        self.encoder_vsn = encoder_vsn
        self.pooling = nn.AdaptiveAvgPool1d((code_size))
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, visual_input_tensor, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask, visual_input_tensor), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask, visual_input_tensor):
        output_trj = self.encoder_trj(self.src_embed(src), src_mask)
        output_vsn = self.encoder_vsn(visual_input_tensor)
        code = torch.cat((output_trj,output_vsn),-1).cuda().double()
        #return self.encoder(self.src_embed(src), src_mask)
        return output_trj + output_vsn

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
