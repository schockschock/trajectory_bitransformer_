import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import sinusoid_encoding_table

class EncoderSelfAttention(nn.Module):
    def __init__(self, device, d_model, d_k, d_v,n_heads, dff=2048, dropout_tf = 0.1, n_module=6) -> None:
        super(EncoderSelfAttention, self).__init__()
        self.encoder = nn.ModuleList([MultiheadAttention(d_model, d_k, d_v, n_heads, dff, dropout_tf)
                                      for _ in range(n_module)])
        self.device = device
    def forward(self, x):
        #positional embedding
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).to(self.device)
        for layer in self.encoder:
            in_encoder = l(in_encoder, in_encoder, in_encoder)
        return in_encoder

class EncoderTransformer(nn.Module):
    def __init__(self, device, d_model, d_k, d_v, n_head=8, n_module=6, ff_size=2048, dropout=0.5) -> None:
        super(EncoderTransformer, self).__init__()
        self.device = device
        #Embedding to 512
        self.fc = nn.Linear(2,d_model)
        self.self_attention = EncoderSelfAttention(device,d_model,d_k,d_v,n_head,ff_size,dropout,n_module)
        self.pool = nn.AdaptiveAvgPool2d((8,d_model)) #final pooling
        
    def forward(self, x):
        shape = x.shape
        print(x.size())
        #x = self.features(x)
        x = x.view(shape[0],shape[1],-1)
        print(x.size())
        x = x.double()
        x = self.fc(x)
        x = F.relu(x)
        x = self.self_attention(x)
        print(x.size())
        #x = self.ffn()
        x = self.pool(x).squeeze(dim=1) #final pooling
        return x
        