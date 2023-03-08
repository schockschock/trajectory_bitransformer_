import torch
import torch.nn as nn

#import of attention layer
from .attention import ScaledDotProductAttention

class EncoderLayer(nn.Module):
    """
    Encoder Layer du papier "Multimodal Transformer", cf fig 2 du papier
    """
    def __init__(self,device,d_model, d_k, d_v, n_head, code_size=1024, dff=2048, dropout_transformer=.1, n_module=3):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = ScaledDotProductAttention(d_model, d_k, d_v, n_head)
        
        #Celle du haut dans le papier
        self.cross_attention_features = ScaledDotProductAttention(d_model, d_k, d_v, n_head)
        #Celle du bas dans le papier
        self.cross_attention_coords = ScaledDotProductAttention(d_model, d_k, d_v, n_head)
        
        self.ffn_coords = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=False), nn.Dropout(p=dropout_transformer),nn.Linear(dff, code_size)])
        
        
        self.ffn_features = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=False), nn.Dropout(p=dropout_transformer),nn.Linear(dff, code_size)])
        
#         self.ffn_coords = nn.Linear(d_model, code_size)
#         self.ffn_features = nn.Linear(d_model, code_size)
        
        self.dropout = nn.Dropout(p=dropout_transformer)
        self.layer_norm = nn.LayerNorm(code_size)
        self.relu = nn.ReLU()
        
    def forward(self, in_encoder_coords, in_encoder_features):
        """
        Encoder Layer return 2 output
        
        """
        
        coords_self_att = self.self_attention(in_encoder_coords,in_encoder_coords,in_encoder_coords)
        
        cross_attention_features = self.cross_attention_features(in_encoder_features,coords_self_att,coords_self_att)
        
        cross_attention_coords = self.cross_attention_coords(coords_self_att,in_encoder_features,in_encoder_features)
        
        out_coords = self.relu(self.ffn_coords(cross_attention_coords))
        
        out_features = self.relu(self.ffn_features(cross_attention_features))
        
        out_coords = self.dropout(out_coords)
        out_features = self.dropout(out_features) 
        
        return self.layer_norm(out_coords), self.layer_norm(out_features)