import math
import torch.nn as nn
import torch
from .functional import clones, attention
from torch.nn.functional import relu
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, flag='default', cross_attention=False):
        """
        Implements Figure 2
        """

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        if flag == 'default':
            return self.linears[-1](x)
        else:
            return self.linears[-1](x), self.attn


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(relu(self.w_1(x))))


class PositionalDecoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalDecoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe=pe.repeat(128,1,1)
        self.register_buffer('pe', pe)
        self.query_embed = nn.Embedding(45, d_model)
        self.lut = nn.Linear(65, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print('xxx',x.shape,self.pe.shape,self.pe[:, :x.size(1)].shape)
        # if encode:
        # print(x.shape)
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        # print('input',x.shape,self.pe[:,:x.size(1)].shape)
        posEmbed = self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        x = torch.cat((Variable(posEmbed, requires_grad=False), x), axis=-1)
        # x=self.lut(x)
        # print(x.shape)
        # print('dec_inputshape',x.shape)
        # exit()

        # else:
        #     query_embed = self.query_embed.unsqueeze(0)
        #     x=x+Variable(query_embed, requires_grad=True)
        # print('shapeeee',self.pe[:, :x.size(1)].shape,x.shape)
        # exit()
        # else:
        #     query_embed = self.query_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.query_embed = nn.Embedding(45, d_model)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)
