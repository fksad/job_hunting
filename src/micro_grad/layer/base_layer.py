# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:19
from typing import List

from src import BaseModule
from src.micro_grad.neuron.base_neuron import BaseNeuron


class BaseLayer(BaseModule):
    def __init__(self, in_dim: int, out_dim: int, neuron_class: BaseNeuron.__class__):
        super().__init__()
        self._in_dim: int = in_dim
        self._out_dim: int = out_dim
        self._neuron_list: List[BaseNeuron] = [neuron_class(in_dim) for _ in range(out_dim)]

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def parameters(self):
        return [p for neuron in self._neuron_list for p in neuron.parameters]

    def forward(self, x):
        out = [neuron.forward(x) for neuron in self._neuron_list]
        return out

    def backward(self):
        for neuron in self._neuron_list:
            neuron.backward()
