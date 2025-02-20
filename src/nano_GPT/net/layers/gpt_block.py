# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 17:04
from torch import nn

from src.micro_grad.mlp import MLP
from src.nano_GPT.config.model_config import ModelConfig
from src.nano_GPT.net.layers.attention.multi_head_attention import MultiHeadAttention


class GPTBlock(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(GPTBlock).__init__()
        self.multi_attention_layer = MultiHeadAttention(model_config)
        self.mlp_layer = MLP(model_config)
        self.ln1 = nn.LayerNorm(model_config.embed_size)
        self.ln2 = nn.LayerNorm(model_config.embed_size)

    def forward(self, x):
        x = x + self.att(self.ln1(x))  # residual connection
        x = x + self.ffn(self.ln2(x))
        return x
