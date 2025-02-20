# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/5 11:57
from typing import List, Tuple

import torch
from torch.nn import functional as F

from src.language_model.base_model import BaseModel


class BiGramMLP(BaseModel):
    def __init__(self, window_size: int, embed_dim: int, hidden_dim: int, regular: float):
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.parameters = []
        self.regular = regular
        self.voc_size = len(self._atoi)

    def train(self,
        train_data: List[int],
        train_label: List[int],
        epochs: int=100,
        learning_rate_decay: list[float]=None,
        batch_size: int=32,
    ) -> None:
        train_data, train_label = self._gen_gram(train_data, train_label)
        train_data = torch.tensor(train_data, dtype=torch.int32)
        train_label = torch.tensor(train_label, dtype=torch.int32)
        learning_rate = self._gen_learning_rate(learning_rate_decay, epochs)
        self.embed_weights = torch.rand((self.voc_size, self.embed_dim), dtype=torch.float32)
        self.hidden_weights = torch.rand((self.embed_dim * self.window_size, self.hidden_dim), dtype=torch.float32)
        self.hidden_bias = torch.rand(self.hidden_dim, dtype=torch.float32)
        self.softmax_weights = torch.rand((self.hidden_dim, self.voc_size), dtype=torch.float32)
        self.softmax_bias = torch.rand(self.voc_size, dtype=torch.float32)
        self.parameters = [self.embed_weights,
                           self.hidden_bias, self.hidden_weights,
                           self.softmax_weights, self.softmax_bias]
        for param in self.parameters:
            param.requires_grad = True

        for epoch in range(epochs):
            train_data_batch, train_label_batch = self._draw_batch(batch_size, train_data, train_label)
            logit_matrix = self._forward(train_data_batch)
            regularity = sum([(p**2).sum() for p in self.parameters])
            loss = F.cross_entropy(logit_matrix, train_label_batch.long()) + self.regular * regularity
            for param in self.parameters:
                param.grad = None
            loss.backward()
            for param in self.parameters:
                param.data -= 10**learning_rate[epoch] * param.grad
            if (epoch + 1) % 500 == 0:
                print(f'epoch {epoch + 1}, loss: {loss.item()}, cur_learning_rate: {10**learning_rate[epoch]}')

    def predict(self, test_data: str) -> float:
        char_idx = [self._atoi[char] for char in test_data]
        test_data, _ = self._gen_gram(char_idx)
        test_data = torch.tensor(test_data, dtype=torch.int64)
        logit = self._forward(test_data)
        exp_logit = logit.exp()
        prob_matrix = exp_logit / exp_logit.sum(dim=1, keepdim=True)
        intput_len = len(test_data) - 1
        loss_prob_vector = prob_matrix[torch.arange(intput_len), char_idx[self.window_size:self.window_size + intput_len]]
        loss = -loss_prob_vector.log().mean()
        return loss.item()

    def _gen_gram(self, data, label=None):
        assert len(data) > self.window_size, 'input data size must be larger than window_size'
        data = [data[i:i+self.window_size] for i in range(0, len(data) - self.window_size + 1)]
        if label is not None:
            label = torch.tensor(label[self.window_size-1:], dtype=torch.int64)
        return data, label

    def _draw_batch(
        self,
        batch_size: int,
        train_data: torch.tensor,
        train_label: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        sample_idx = torch.randint(0, train_data.shape[0], (batch_size, ))
        train_data_batch = train_data[sample_idx]
        train_label_batch = train_label[sample_idx]
        return train_data_batch, train_label_batch

    def _forward(self, x: torch.tensor) -> torch.tensor:
        input_matrix = self.embed_weights[x]
        data_matrix = torch.tanh(input_matrix.view(-1, self.window_size * self.embed_dim) @ self.hidden_weights + self.hidden_bias)
        logit_matrix = data_matrix @ self.softmax_weights + self.softmax_bias
        return logit_matrix

    def _gen_learning_rate(self, learning_rate_decay: List[float], epochs: int) -> List[float]:
        learning_rate_list = []
        learning_rate_count = len(learning_rate_decay)
        seg_len = epochs // learning_rate_count
        for i in range(learning_rate_count - 1):
            learning_rate_list.extend([learning_rate_decay[i]]*seg_len)
        learning_rate_list.extend([learning_rate_decay[-1]]*(epochs - len(learning_rate_list)))
        return learning_rate_list
