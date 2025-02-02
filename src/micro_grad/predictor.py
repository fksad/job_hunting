# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 19:45
from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, predict_data):
        raise NotImplementedError()

class ClassificationPredictor(BasePredictor):
    def predict(self, predict_data):
        class_label_list = []
        for cur_data in predict_data:
            pred_data = self.model.forward(cur_data)
            class_label = np.argmax([p.data for p in pred_data])
            class_label_list.append(class_label)
        return class_label_list
