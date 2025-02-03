# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 15:08
from typing import List

from src import BaseModule
from src.micro_grad.layer import LinearLayer
from src.micro_grad.layer.relu_layer import ReluLayer
from src.micro_grad.layer.sigmoid_layer import SigmoidLayer


class MLP(BaseModule):
    def __init__(self, in_dim: int, hidden_dim_list: List[int], out_dim: int):
        dim_list = [in_dim] + hidden_dim_list + [out_dim]
        self._layer_list = []
        for layer_idx, (cur_in_dim, cur_out_dim) in enumerate(zip(dim_list[:-1], dim_list[1:])):
            if layer_idx == 0:
                cur_layer = ReluLayer(cur_in_dim, cur_out_dim)
            elif layer_idx == len(dim_list) - 2:
                cur_layer = LinearLayer(cur_in_dim, cur_out_dim)
            else:
                cur_layer = LinearLayer(cur_in_dim, cur_out_dim)
            self._layer_list.append(cur_layer)

    @property
    def parameters(self):
        return [p for layer in self._layer_list for p in layer.parameters]

    def forward(self, x):
        for idx, layer in enumerate(self._layer_list):
            x = layer.forward(x)
        return x

    def backward(self):
        self.clear_grad()
        for p in self.parameters:
            p.backward()
