# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:52
from typing import List

from src.micro_grad.loss.base_loss import BaseLoss
from src.micro_grad.value import Value


class RMSELoss(BaseLoss):
    def __init__(self):
        BaseLoss.__init__(self)

    def __call__(self, y_pred: List[List[Value]], y_true: list[float]) -> Value:
        loss = 0
        for y, yp in zip(y_true, y_pred):
            residual = y - yp[0]
            loss += residual ** 2
        loss /= len(y_true)
        loss = loss ** 0.5
        return loss
