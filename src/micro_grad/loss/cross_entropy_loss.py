# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 20:53
from typing import List

from src.micro_grad.loss.base_loss import BaseLoss
from src.micro_grad.value import Value


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        BaseLoss.__init__(self)

    def __call__(self, y_pred: List[List[Value]], y_true: List[List[int]]) -> Value:
        loss = 0
        for y, yp in zip(y_true, y_pred):
            y_logit_list = [y.exp() for y in yp]
            sum_y_logit = sum(y_logit_list)
            y_logit_list = [y / sum_y_logit for y in y_logit_list]
            loss += -sum([y_t * y_logit.log() for y_t, y_logit in zip(y, y_logit_list)])
        loss /= len(y_true)
        return loss
