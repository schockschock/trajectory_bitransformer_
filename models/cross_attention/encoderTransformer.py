import torch
import torch.nn as nn

from .encoderLayer import EncoderLayer
from .embedding import sinusoid_encoding_table


class EncoderTransformer(nn.Module):
    """
    Encoder Transformer adapted from the MTN Paper.
    It only takes as input the (x,y) coordinates as in our case the camera is static (the MTN was used for a dynamic "ego-car" context)
    """

    def __init__(self, device, code_size, d_k, d_v, n_head=8, n_module=3, ff_size=2048, dropout1d=0.5, feature_size=512):
        super(EncoderTransformer, self).__init__()
        self.device = device

        # Embedding
        self.fc_coords = nn.Linear(2, code_size)
        self.fc_features = nn.Linear(feature_size, code_size)

        # n_module layers of encoding
        self.encoder = nn.ModuleList([EncoderLayer(device, code_size, d_k, d_v, n_head, dff=ff_size, dropout_transformer=dropout1d, code_size=code_size)
                                      for _ in range(n_module)])

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((8, code_size))  # final pooling
        self.dropout = nn.Dropout(dropout1d)

    def forward(self, coords, features):
        """
        coords : sequence of coordinates
        features : sequence of features extracted from a ResNet

        """
        # Embedding + positional embedding of coordinates
        coords = self.relu(self.fc_coords(coords))
        in_encoder_coords = coords + sinusoid_encoding_table(
            coords.shape[1], coords.shape[2]).expand(coords.shape).to(self.device)

        # Embedding + positional embedding of features
        in_encoder_features = self.relu(self.fc_features(features))

        # Encoding
        for layer in self.encoder:
            out_enc_coords, out_enc_features = layer(
                in_encoder_coords, in_encoder_features)

        code = (out_enc_coords+out_enc_features).to(device).double()

        return code
