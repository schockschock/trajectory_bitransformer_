import torch
import torch.nn as nn
from .functional import sinusoid_encoding_table
from .attention import MultiHeadAttention

class EncoderSelfAttention(nn.Module):
    def __init__(self,device,d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, n_module=6):
        super(EncoderSelfAttention, self).__init__()
        
        
        self.encoder = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, n_head, dff, dropout_transformer)
                                      for _ in range(n_module)])
        self.device = device
    def forward(self, x):
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).to(self.device)
        for l in self.encoder:
            in_encoder = l(in_encoder, in_encoder, in_encoder)
        return in_encoder

class EncoderTransformer(nn.Module):
    """
    Encoder Transformer adapted from the MTN Paper.
    It only takes as input the (x,y) coordinates as in our case the camera is static (the MTN was used for a dynamic "ego-car" context)
    """
    def __init__(self,device,d_model,d_k, d_v,n_head=8,n_module=6,ff_size=1024,dropout1d=0.5, batch_size=1):
        super(EncoderTransformer, self).__init__()
        self.device = device
        self.fc = nn.Linear(2,d_model)
        self.self_attention = EncoderSelfAttention(device,d_model,d_k,d_v,n_head=n_head,dff=ff_size,dropout_transformer=dropout1d,n_module=n_module)
        self.relu =  nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8,d_model)) #final pooling

    def forward(self, x):
        shape = x.shape
        #print(x.size())
        #x = x.to(device)

        x = x.view(shape[0],shape[1],-1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.self_attention(x)
        x = self.pool(x)
        x = x.squeeze(dim=1) #final pooling
        return x
        