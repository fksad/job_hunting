# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:26
from src.micro_grad.layer.base_layer import BaseLayer
from src.micro_grad.neuron.sigmoid_neuron import SigmoidNeuron


class SigmoidLayer(BaseLayer):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, SigmoidNeuron)
