# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/1/26 16:16
from torch import nn

from src.gpt_demo.config import GPTConfig


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        nn.Module.__init__(self)
        self.key = nn.Linear(config.n_embd, config.n_head)
