import torch
import torch.nn as nn

from .cross_attention.encoderTransformer import EncoderTransformer
from .scratch_transformer.decoder import TransformerDecoder
from .resnet.feature_extraction import features_extraction, Resnet


class Seq2SeqCA(nn.Module):
    """
    SEQ2SEQ MODEL USING THE CROSS ATTENTION MECANISM TO ENCODE BOTH COORDS AND RESNET FEATURES AT THE SAME TIME
    """

    def __init__(self, device, embed_size, code_size=512, target_size=12, dropout_val=0.2, batch_size=1, conv_model=Resnet()):
        super(Seq2SeqCA, self).__init__()

        self.feature_size = 512
        torch.cuda.empty_cache()
        self.encoder = EncoderTransformer(
            device, code_size, 64, 64, dropout1d=dropout_val)  # EncoderTransformer
        # self.encoder.apply(init_weights)

        self.decoder = TransformerDecoder(
            target_size, embed_dim=code_size, seq_len=12, num_layers=2, expansion_factor=4, n_heads=8)
        # self.decoder.apply(init_weights)

        self.features_ex = features_extraction(conv_model, in_planes=3)

        # add a pooling (to have the same shape)
        self.pooling = nn.AdaptiveAvgPool1d((code_size))

        self.code_pooling = nn.AdaptiveAvgPool2d((target_size, code_size))

        if device.type == 'cuda':
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, input_tensor, target_tensor, features):
        batch_size = int(input_tensor.size(0))

        # Les features ont été introduites directement dans le dataset
        #features = self.features_ex(visual_input_tensor)

        features = features.view((batch_size, 8, -1))  # (bs, 8, 512)

        encoder_output = self.encoder(
            input_tensor, features)  # (bs,8,code_size)

        # start_point
#         start_point     = (input_tensor[:,0,:]).to(device).clone().detach()

#         if startpoint_mode=="on":
#             input_tensor[:,0,:]    = 0

#         visual_initial_vsn          = self.vsn_module(visual_input_tensor)
#         visual_initial_vsn          = self.pooling(visual_initial_vsn) #pooling qu'on a ajouté

#         print("#######")
#         print(f"encoder_output size : {encoder_output.size()}")
#         print(f"visual_initial_vsn size : {visual_initial_vsn.size()}")
#         print("#######")

        trg_mask = self.make_trg_mask(target_tensor)
        #print(f"target_tensor : {target_tensor.size()}")

        code_seq_12 = self.code_pooling(encoder_output)
        decoder_output = self.decoder(target_tensor, code_seq_12, trg_mask)

        return decoder_output
