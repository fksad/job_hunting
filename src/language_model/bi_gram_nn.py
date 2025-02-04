# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/4 11:08
from typing import List

import torch
from torch.functional import F


class BiGramNN:
    def __init__(self, class_dim=27):
        self._weights = None
        self._bias = None
        self._class_dim = class_dim

    def train(self, train_set: List[int], train_label: List[int], epochs: int=100, learning_rate: float=0.01) -> None:
        g = torch.Generator().manual_seed(2147483647)
        self._weights = torch.rand((self._class_dim, self._class_dim), requires_grad=True, dtype=torch.float, generator=g)
        for i in range(epochs):
            self._weights.grad = None
            loss = self._forward(train_set, train_label)
            loss.backward()

            self._weights.data -= learning_rate * self._weights.grad
            if i % 5 == 0:
                print(f'Epoch {i} loss: {loss.item()}')

    def predict(self, test_str: str) -> float:
        char_idx_list = [self._char_to_i(char) for char in test_str]
        loss = self._forward(char_idx_list[:-1], char_idx_list[1:])
        return loss.item()

    def _forward(self, train_set, train_label):
        one_hot_matrix = F.one_hot(torch.tensor(train_set), num_classes=self._class_dim).float()
        logits = one_hot_matrix @ self._weights
        exp_logits = logits.exp()
        prob_matrix = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        loss_prob_vector = prob_matrix[torch.arange(len(train_label)),torch.tensor(train_label)]
        loss =  -loss_prob_vector.log().mean()
        return loss

    @staticmethod
    def _char_to_i(char):
        if char != '.':
            return ord(char) - 97
        else:
            return 26