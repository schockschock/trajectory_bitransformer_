import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8) -> None:
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        # 512/8 = 64  . each key,query, value will be of 64d
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        # key querry and value matrixes
        # single key matrix for all 8 keys #512x512
        self.querry_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)

        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)

    # batch_size x sequence_length x embedding_dim    # 32 x 8 x 512
    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # 32x8x512
        # batch_size x sequence_length x n_heads x single_head_dim = (32x8x8x64)
        key = key.view(batch_size, seq_length,
                       self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query,
                           self.n_heads, self.single_head_dim)  # (32x8x8x64)
        value = value.view(batch_size, seq_length, self.n_heads,
                           self.single_head_dim)  # (32x8x8x64)

        k = self.key_matrix(key)       # (32x8x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 8 x 64)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention
        # adjust key for matrix multiplication
        # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 8)
        k_adjusted = k.transpose(-1, -2)
        # (32 x 8 x 8 x 64) x (32 x 8 x 64 x 8) = #(32x8x8x8)
        product = torch.matmul(q, k_adjusted)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by sqaure root of key dimension
        product = product / math.sqrt(self.single_head_dim)

        # applying softmax
        # (32x8x 8x 10) x (32 x 8 x 8 x 64) = (32 x 8 x 8 x 64)
        scores = F.softmax(product, dim=-1)

        # concatenate the output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                          self.single_head_dim*self.n_heads)  # (32x8x8x64) -> (32x8x8x64)  -> (32,8,512)
        output = self.out(concat)  # (32,8,512)

        return output
