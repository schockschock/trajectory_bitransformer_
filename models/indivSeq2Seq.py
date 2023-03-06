import torch
import torch.nn as nn
from .encoders.trajectory_encoder import EncoderTransformer
from .encoders.resnet_encoder import _GestureTransformer
from .scratch_transformer.decoder import TransformerDecoder


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


class Seq2Seq(nn.Module):
    def __init__(self, device, code_size, target_size, dropout=0.1) -> None:
        super(Seq2Seq, self).__init__()

        # encoder of the coordinates
        self.encoder = EncoderTransformer(
            device=device, d_model=code_size, d_k=64, d_v=64, dropout=dropout)
        self.enc_embed_dim = code_size*2

        # encoder of the images
        # _GestureTransformer(partie vision)
        self.vsn_module = _GestureTransformer(
            device, input_dim=code_size, dropout=dropout)
        self.vsn_module.apply(init_weights)

        # decoder transformer
        self.decoder = TransformerDecoder(target_size, self.enc_embed_dim)

        # pooling layers
        # add a pooling (to have the same shape)
        self.pooling = nn.AdaptiveAvgPool1d((code_size))

        self.code_pooling = nn.AdaptiveAvgPool2d(
            (target_size, embed_dim))  # ?

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size = trg.shape[0]
        trg_len = 12
        #batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, input_tensor, target_tensor, visual_input_tensor, batch_size, train_mode):
        batch_size = int(input_tensor.size(0))
        print("#####Encoding coordinates#####")
        encoder_output = self.encoder(input_tensor)
        print("#####Encoding images#####")
        visual_initial_vsn = self.vsn_module(visual_input_tensor)
        visual_initial_vsn = self.pooling(
            visual_initial_vsn)  # pooling qu'on a ajouté
        print("#######")
        print(f"encoder_output size : {encoder_output.size()}")
        print(f"visual_initial_vsn size : {visual_initial_vsn.size()}")
        print("#######")
        # Creattion du code (concatenation du transformer coordonnées et transformer resnet)
        code = torch.cat((encoder_output, visual_initial_vsn), -1)
        # pooling qu'on a rajouté pour que la taille du code soit la meme que la target
        code = self.code_pooling(code)

        print("#######")
        print(f"size of encoder output for decoder input :{code.size()}")
        print("#######")
        trg_mask = self.make_trg_mask(target_tensor)
        print(f"target_tensor : {target_tensor.size()}")
        decoder_output = self.decoder(target_tensor, code, trg_mask)
        print(decoder_output.size())
        return decoder_output