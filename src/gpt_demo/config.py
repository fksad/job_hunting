# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/1/26 16:17
from pydantic import BaseModel


class GPTConfig(BaseModel):
    max_seq_len: int = 512   # 文本的最大长度
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hidden_size, 这里我同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
    vocab_size: int = 50257

