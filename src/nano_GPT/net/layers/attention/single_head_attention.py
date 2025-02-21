# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 15:53
import torch
from torch import nn
from torch.nn import functional as F

from src.nano_GPT.config.model_config import ModelConfig


class SingleHeadAttention(nn.Module):
    def __init__(self, model_config: ModelConfig):
        nn.Module.__init__(self)
        self.seq_len = model_config.seq_len
        self.head_size = model_config.head_size
        self.embed_size = model_config.embed_size
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.dropout = nn.Dropout(model_config.dropout_ratio)
        self.register_buffer('mask_shape', torch.tril(torch.ones([self.seq_len, self.seq_len])))  # do not calculate grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)  # B, T, H
        k = self.key(x)  # B, T, H
        v = self.value(x)  # B, T, H
        weight = q @ k.transpose(-2, -1) * C**.5  # B, T, T
        weight = weight.masked_fill(self.mask_shape==0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        attention = weight @ v
        return attention
