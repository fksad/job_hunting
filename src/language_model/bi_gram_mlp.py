# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/5 11:57
from typing import List, Tuple

import torch
from torch.distributions.utils import logits_to_probs
from torch.nn import functional as F

from src.language_model.base_model import BaseModel


class BiGramMLP(BaseModel):
    def __init__(self, embed_dim: int, hidden_dim: int, regular: float):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.parameters = []
        self.regular = regular
        self.voc_size = len(self._atoi)

    def train(self,
        train_data: List[int],
        train_label: List[int],
        epochs: int=100,
        learning_rate: float=.01,
        batch_size: int=32,
    ) -> None:
        train_data = torch.tensor(train_data, dtype=torch.int32)
        train_label = torch.tensor(train_label, dtype=torch.int32)
        self.embed_weights = torch.rand((self.voc_size, self.embed_dim), dtype=torch.float32)
        self.hidden_weights = torch.rand((self.embed_dim, self.hidden_dim), dtype=torch.float32)
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
                param.data -= learning_rate * param.grad
            if epoch % 100 == 0:
                print(f'epoch {epoch}, loss: {loss.item()}')

    def predict(self, test_data: str) -> float:
        char_idx = torch.tensor([self._atoi[char] for char in test_data], dtype=torch.int64)
        logit = self._forward(char_idx)
        exp_logit = logit.exp()
        prob_matrix = exp_logit / exp_logit.sum(dim=1, keepdim=True)
        loss_prob_vector = prob_matrix[torch.arange(len(char_idx) - 1), char_idx[1:]]
        loss = -loss_prob_vector.log().mean()
        return loss.item()

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
        data_matrix = torch.tanh(input_matrix @ self.hidden_weights + self.hidden_bias)
        logit_matrix = data_matrix @ self.softmax_weights + self.softmax_bias
        return logit_matrix
