import torch
import torch.nn as nn

from .encoderTransformer import EncoderTransformer
from ..scratch_transformer.decoder import TransformerDecoder

class Seq2SeqCA(nn.Module):
    """
    SEQ2SEQ MODEL USING THE CROSS ATTENTION MECANISM TO ENCODE BOTH COORDS AND RESNET FEATURES AT THE SAME TIME
    """
    def __init__(self,device, embed_size, code_size=512, target_size = 12, dropout_val=dropout_val, batch_size=1, conv_model=Resnet()):
        super(Seq2SeqCA, self).__init__()
        
        self.feature_size = 512
        torch.cuda.empty_cache()
        self.encoder = EncoderTransformer(device,code_size,64,64,dropout1d=dropout_val) #EncoderTransformer
        
        
        self.decoder = TransformerDecoder(target_size, embed_dim=code_size, seq_len=12, num_layers=2, expansion_factor=4, n_heads=8) 
        #self.decoder.apply(init_weights)
        
        self.feature_size = 512
        self.features_ex = features_extraction(conv_model,in_planes=3)

        self.pooling = nn.AdaptiveAvgPool1d((code_size)) # add a pooling (to have the same shape)
        
        self.code_pooling = nn.AdaptiveAvgPool2d((target_size,code_size))
        
        if device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()  
            
    def forward(self,input_tensor, target_tensor, features):
        batch_size      = int(input_tensor.size(0))

        #Les features ont été introduites directement dans le dataset
        #features = self.features_ex(visual_input_tensor)

        features = features.view((batch_size,8,-1)) #(bs, 8, 512)

        encoder_output =  self.encoder(input_tensor, features) #(bs,8,code_size)
        
        trg_mask = self.make_trg_mask(target_tensor)
        #print(f"target_tensor : {target_tensor.size()}")
        
        code_seq_12 = self.code_pooling(encoder_output)
        decoder_output = self.decoder(target_tensor,code_seq_12,trg_mask)

        return decoder_output
    
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
        return trg_mask.to(device)