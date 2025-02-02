# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:24
from src.micro_grad.layer.base_layer import BaseLayer
from src.micro_grad.neuron.linear_neuron import LinearNeuron


class LinearLayer(BaseLayer):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, LinearNeuron)
