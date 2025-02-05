# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/5 15:27
import string
from abc import ABC, abstractmethod
from typing import Dict


class BaseModel(ABC):
    _special_tokens = '.'
    _atoi: Dict[str, int] = {i: idx for idx, i in enumerate(string.ascii_lowercase)}
    _itoa: Dict[int, str] = {idx: i for i, idx in _atoi.items()}
    _atoi[_special_tokens] = len(_itoa)
    _itoa[len(_itoa)] = _special_tokens

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: str) -> float:
        raise NotImplementedError
