# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:14
from abc import ABC, abstractmethod

from src import BaseModule


class BaseNeuron(BaseModule):
    def __init__(self, in_dim):
        self._in_dim = in_dim

    @property
    def in_dim(self):
        return self._in_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        for param in self.parameters:
            param.backward()
