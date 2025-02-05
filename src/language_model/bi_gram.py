# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/3 20:20
from typing import List, Dict
import string

import torch

from src.language_model.base_model import BaseModel


class BiGram(BaseModel):


    def __init__(self):
        self._output_count_matrix = torch.ones(len(self._itoa), len(self._itoa))
        self._prob_matrix = torch.zeros(len(self._itoa), len(self._itoa))

    def train(self, corpus: List[str]) -> None:
        for line in corpus:
            padded_line = self._special_tokens + line + self._special_tokens
            for char_1, char_2 in zip(padded_line, padded_line[1:]):
                self._output_count_matrix[self._atoi[char_1], self._atoi[char_2]] += 1
        self._prob_matrix = self._output_count_matrix / self._output_count_matrix.sum(dim=1, keepdim=True)

    def predict(self, test_data: str) -> float:
        log_likelihood = 0
        n = 0
        padded_str = self._special_tokens + test_data + self._special_tokens
        for char_1, char_2 in zip(padded_str, padded_str[1:]):
            prob = self._prob_matrix[self._atoi[char_1], self._atoi[char_2]]
            log_likelihood += torch.log(prob)
            n += 1
        return -log_likelihood / n

    def generate(self) -> str:
        char_list = []
        cur_char = self._special_tokens
        while True:
            cur_idx = self._atoi[cur_char]
            cur_prob = self._prob_matrix[cur_idx]
            next_char_idx = torch.multinomial(cur_prob, 1, replacement=True)
            cur_char = self._itoa[next_char_idx.item()]
            if cur_char == self._special_tokens:
                break
            char_list.append(cur_char)
        return ''.join(char_list)
