# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 16:43
import torch
from torch import nn

from src.nano_GPT.config.model_config import ModelConfig
from src.nano_GPT.net.layers.attention.single_head_attention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(MultiHeadAttention).__init__()
        self.attention_head_list = [SingleHeadAttention(model_config) for _ in range(model_config.n_head)]

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_head_list], dim=-1)
        return out
