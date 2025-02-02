# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:14
import numpy as np

from src.micro_grad.neuron.base_neuron import BaseNeuron
from src.micro_grad.value import Value


class LinearNeuron(BaseNeuron):
    def __init__(self, in_dim):
        super().__init__(in_dim)
        self.weights = [Value(np.random.rand()) for _ in range(in_dim)]
        self.bias = Value(np.random.rand())

    @property
    def parameters(self):
        return self.weights + [self.bias]

    def forward(self, x):
        out = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return out

