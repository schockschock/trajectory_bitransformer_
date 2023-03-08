import torch
import torch.nn as nn
from .encoders.trajectory_encoder import EncoderTransformer
from .encoders.resnet_encoder import _GestureTransformer
from .scratch_transformer.decoder import TransformerDecoder


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


class Seq2Seq(nn.Module):
    def __init__(self,device, embed_size, code_size=512, target_size = 12, dropout_val=0.2, batch_size=1):
        super(Seq2Seq, self).__init__()

        self.device = device

        torch.cuda.empty_cache()
        self.encoder = EncoderTransformer(device,code_size,64,64,dropout1d=dropout_val, batch_size=batch_size) #EncoderTransformer
        #self.encoder.apply(init_weights)
        
        #vision transformer for video inputs
        self.vsn_module = _GestureTransformer(device,input_dim=code_size,dropout1d=dropout_val) #_GestureTransformer(partie vision)                   
        self.vsn_module.apply(init_weights)
        
        #Transformer decoder
        embed_dim = 2*code_size #Embedding of decoder is the size of the concatenation of two codes
        self.decoder = TransformerDecoder(target_size, embed_dim=embed_dim, seq_len=12, num_layers=2, expansion_factor=4, n_heads=8) 
        
        #Pooling layers
        self.pooling = nn.AdaptiveAvgPool1d((code_size)) # applied to vision transformer output
        self.code_pooling = nn.AdaptiveAvgPool2d((target_size,embed_dim))
        
        if device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()
            self.vsn_module.cuda()   
            
    def forward(self,input_tensor, target_tensor, visual_input_tensor, batch_size, train_mode):
        batch_size      = int(input_tensor.size(0))
        
        encoder_output =  self.encoder(input_tensor) #(bs,8,code_size)         
        visual_initial_vsn          = self.vsn_module(visual_input_tensor)
        visual_initial_vsn          = self.pooling(visual_initial_vsn) #pooling qu'on a ajouté
 
        
        #Creation du code (concatenation du transformer coordonnées et transformer resnet)
        code = torch.cat((encoder_output,visual_initial_vsn),-1).to(self.device).double()
        
        code = self.code_pooling(code) #pooling qu'on a rajouté pour que la taille du code soit la meme que la target
        trg_mask = self.make_trg_mask(target_tensor)
        decoder_output = self.decoder(target_tensor,code,trg_mask)

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
        return trg_mask.to(self.device)