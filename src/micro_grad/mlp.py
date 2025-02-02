# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 15:08
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.micro_grad.value import Value


class BaseModule(ABC):
    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError()

    def clear_grad(self):
        for param in self.parameters:
            param.grad = 0


class LinearNeuron(BaseModule):
    def __init__(self, in_dim):
        self.weights = [Value(np.random.rand()) for _ in range(in_dim)]
        self.bias = Value(np.random.rand())
        self.in_dim = in_dim

    @property
    def parameters(self):
        return self.weights + [self.bias]

    def forward(self, x):
        out = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        out = out.relu()
        return out

    def backward(self):
        for param in self.parameters:
            param.backward()


class LinearLayer(BaseModule):
    def __init__(self, in_dim, out_dim):
        self.neuron_list = [LinearNeuron(in_dim) for _ in range(out_dim)]

    @property
    def parameters(self):
        return [p for neuron in self.neuron_list for p in neuron.parameters]

    def forward(self, x):
        out = [neuron.forward(x) for neuron in self.neuron_list]
        return out

    def backward(self):
        for neuron in self.neuron_list:
            neuron.backward()


class MLP(BaseModule):
    def __init__(self, in_dim: int, hidden_dim_list: List[int], out_dim: int):
        dim_list = [in_dim] + hidden_dim_list + [out_dim]
        self.layer_list = [LinearLayer(dim_list[i], dim_list[i+1]) for i in range(len(dim_list) - 1)]

    @property
    def parameters(self):
        return [p for layer in self.layer_list for p in layer.parameters]

    def forward(self, x):
        for layer in self.layer_list:
            x = layer.forward(x)
        return x

    def backward(self):
        self.clear_grad()
        for p in self.parameters:
            p.backward()
