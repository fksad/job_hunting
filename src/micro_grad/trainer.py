# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 15:53
from abc import ABC

import numpy as np

from src.micro_grad.loss import RMSELoss, BaseLoss
from src.micro_grad.mlp import MLP


class BaseTrainer(ABC):
    def __init__(self, model: MLP, learning_rate: float = .1, epochs: int = 100, early_stopping: bool = False,
                 loss: BaseLoss = None):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.loss = loss or RMSELoss()

class Trainer(BaseTrainer):

    def train(self, train_set, stop_loss=0, show_epoch=False):
        for epoch in range(self.epochs):
            train_data, train_labels = train_set
            pred_data_list = []
            for cur_data in train_data:
                pred_data = self.model.forward(cur_data)
                pred_data_list.append(pred_data)
            loss = self.loss(pred_data_list, train_labels)

            if epoch % 5 == 0:
                acc = ''
                if show_epoch:
                    label_list = [np.argmax([v.data for v in pred_data]) for pred_data in pred_data_list]
                    acc = [np.argmax(train_label)==label for train_label, label in zip(train_labels, label_list)]
                    acc = sum(acc) / len(acc)
                print(f'epoch: {epoch}, loss: {loss}, acc: {acc}')
            if self.early_stopping and loss.data < stop_loss:
                break
            else:
                self.model.clear_grad()
                loss.backward()
                for param in self.model.parameters:
                    param.data -= self.learning_rate * param.grad
