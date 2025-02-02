# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:52
from abc import ABC, abstractmethod
from math import inf
from typing import List

from src.micro_grad.value import Value


class BaseLoss(ABC):
    def __init__(self):
        self.value = inf

    @abstractmethod
    def __call__(self, y_pred: List[List[Value]], y_true) -> Value:
        raise NotImplementedError
