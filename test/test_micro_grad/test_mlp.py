# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 16:11
import unittest

import numpy as np

from src.micro_grad.loss import RMSELoss, CrossEntropyLoss
from src.micro_grad.predictor import ClassificationPredictor
from src.micro_grad.trainer import Trainer
from src.micro_grad.mlp import MLP
from sklearn.datasets import make_moons, make_blobs
np.random.seed(42)

class MLPTestCase(unittest.TestCase):
    def setUp(self):
        self.sample_size = 100
        self.epochs = 100

    def test_regression(self):
        train_dataset, train_label = make_moons(n_samples=self.sample_size, noise=0.001, random_state=42)
        train_label = train_label * 2 - 1  # make y be -1 or 1
        self._model = MLP(2, [4, 4], 1)
        self._trainer = Trainer(model=self._model, epochs=self.epochs, learning_rate=0.01, loss=RMSELoss())
        self._trainer.train((train_dataset, train_label))

    def test_classification(self):
        class_dim = 3
        train_dataset, train_label = make_blobs(n_samples=self.sample_size, centers=class_dim, n_features=5, random_state=0)
        predict_dataset, gt_label = make_blobs(n_samples=self.sample_size, centers=class_dim, n_features=5, random_state=1)
        train_label = np.eye(class_dim)[train_label]
        self._model = MLP(5, [4, 4], class_dim)
        self._trainer = Trainer(model=self._model, epochs=self.epochs, learning_rate=0.05, loss=CrossEntropyLoss())
        self._trainer.train((train_dataset, train_label), show_epoch=True)
        self._predictor = ClassificationPredictor(self._model)
        predict_label = self._predictor.predict(predict_dataset)
        acc = sum([p_label==gt_label for p_label, gt_label in zip(predict_label, gt_label)]) / len(predict_label)
        print(f'predict acc: {acc}')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
