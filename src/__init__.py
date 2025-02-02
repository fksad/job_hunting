# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/1/26 16:13
from abc import ABC, abstractmethod


class BaseModule(ABC):
    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError()

    def clear_grad(self):
        for param in self.parameters:
            param.grad = 0
