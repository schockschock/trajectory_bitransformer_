import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy
import re
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
import warnings
from torch.autograd import Variable

warnings.simplefilter("ignore")


# Definition of variables
embed_dim = 512
input_size = 8


class Embedding(nn.Module):
    def __init__(self, embed_dim, input_size) -> None:
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(input_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embed_dim)

class PositionalEncoding(nn.Module):
    "Positional encoding function"
    def __init__(self, embed_dim, dropout, max_len=(5000)) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        
        #compute the positional encoding in log space
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,embed_dim,2).float * -(math.log(10000.0)/embed_dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #Variable is now deprecated
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
        

