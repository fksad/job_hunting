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
        self._one_hot_matrix = F.one_hot(torch.tensor(train_set), num_classes=self._class_dim).float()
        for i in range(epochs):
            self._weights.grad = None
            loss = self._gen_neg_log_likelihood(train_label)
            loss.backward()

            self._weights.data -= learning_rate * self._weights.grad
            if i % 5 == 0:
                print(f'Epoch {i} loss: {loss.item()}')

    def predict(self, test_str: str) -> float:
        char_idx_list = [self._char_to_i(char) for char in test_str]
        test_one_hot = F.one_hot(torch.tensor(char_idx_list), num_classes=self._class_dim).float()
        test_logits = test_one_hot @ self._weights
        test_exp_logits = test_logits.exp()
        test_prob = test_exp_logits / test_exp_logits.sum(dim=1, keepdim=True)
        test_loss_prob_vector = test_prob[torch.arange(test_one_hot.shape[0]-1), torch.tensor(char_idx_list[1:])]
        loss = -test_loss_prob_vector.log().mean()
        return loss.item()

    def _gen_neg_log_likelihood(self, train_label):
        logits = self._one_hot_matrix @ self._weights
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