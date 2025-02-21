# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 17:26
from typing import List, Protocol

import ujson as json
import torch
from torch.utils.data import Dataset


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, ids: List[int]) -> List[str]:
        ...


class NanoGPTDataset(Dataset):

    def __init__(self, data_path: str, seq_len: int=512, tokenizer: Tokenizer=None, max_lines: int=1000):
        import tiktoken
        self._data_path = data_path
        self._encoder = tokenizer or tiktoken.get_encoding("gpt2")
        self._seq_len = seq_len
        self._max_lines = max_lines
        self._is_json = data_path.endswith(".json")
        self._eos_token = self._encoder.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        self._encoded_chunk_list = self._load_file()

    def _load_file(self):
        encoded_text_list = []
        encoded_text_chunk_list = []
        with open(self._data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self._max_lines:
                    break
                if self._is_json:
                    text = json.loads(line.strip())['text']
                else:
                    text = line.strip()
                encoded_text = self._encoder.encode(text) + [self._eos_token]
                encoded_text_list += encoded_text
        for i in range(0, len(encoded_text_list), self._seq_len):
            chunk = encoded_text_list[i:i + self._seq_len + 1]  # add label
            if len(chunk) < self._seq_len + 1:
                chunk += [self._eos_token] * (self._seq_len + 1 - len(chunk))
            encoded_text_chunk_list.append(chunk)
        return encoded_text_chunk_list

    def __len__(self):
        return len(self._encoded_chunk_list)

    def __getitem__(self, idx):
        chunk = self._encoded_chunk_list[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self._encoder.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self._encoder.decode(ids)
