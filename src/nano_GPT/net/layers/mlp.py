# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 16:58
from torch import nn

from src.nano_GPT.config.model_config import ModelConfig


class MLP(nn.Module):
    def __init__(self, model_config: ModelConfig):
        nn.Module.__init__(self)
        self.layer = nn.Sequential(
            nn.Linear(model_config.embed_size, model_config.embed_size * 4),
            nn.GELU(),
            nn.Linear(model_config.embed_size * 4, model_config.embed_size),
            nn.Dropout(model_config.dropout_ratio),
        )

    def forward(self, x):
        out = self.layer(x)
        return out
