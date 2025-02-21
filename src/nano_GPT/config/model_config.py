# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 16:53
from typing import Tuple

import torch
from pydantic import BaseModel


class ModelConfig(BaseModel):
    seq_len: int = 8   # 这里其实应该是文本的最大长度（ max_seq_len）
    n_block: int = 2
    n_head: int = 2
    embed_size: int = 16
    head_size: int = embed_size // n_head
    vocab_size: int = 50257
    dropout_ratio: float = 0.1


type GeneralDataLoader = torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor]]

