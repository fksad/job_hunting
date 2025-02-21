# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 16:43
import torch
from torch import nn

from src.nano_GPT.config.model_config import ModelConfig
from src.nano_GPT.net.layers.attention.single_head_attention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_config: ModelConfig):
        nn.Module.__init__(self)
        self.attention_head_list = nn.ModuleList(
            [SingleHeadAttention(model_config) for _ in range(model_config.n_head)]
        )
        self.projection_layer = nn.Linear(model_config.embed_size, model_config.embed_size)
        self.drop = nn.Dropout(model_config.dropout_ratio)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_head_list], dim=-1)
        out = self.projection_layer(out)
        out = self.drop(out)
        return out
