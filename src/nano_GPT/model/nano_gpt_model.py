# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 17:17
import os
import time
from typing import Callable, Tuple

import torch

from src.nano_GPT.config.metric_sequence import MetricSequence
from src.nano_GPT.config.model_config import ModelConfig, GeneralDataLoader
from src.nano_GPT.net.nano_gpt_net import NanoGPTNet

from src.nano_GPT.config.metric import Metric
from src.nano_GPT.model.base_model import BaseModel
from src.nano_GPT.utils import gen_metric


class NanoGPTModel(BaseModel):
    def __init__(self, config: ModelConfig, device: torch.device):
        super(NanoGPTModel).__init__()
        self._model_config = config
        self._device = device
        self._net = NanoGPTNet(config).to(self._device)

    def __getattr__(self, item):
        return getattr(self._net, item)

    def train(self, data_loader: GeneralDataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        start_time = time.time()
        self._net.train()
        metric_list = MetricSequence()
        for train_data, train_labels in data_loader:
            self._net.zero_grad()
            inputs = train_data.to(self._device)
            labels = train_labels.to(self._device)
            outputs, loss = self._net(inputs)
            metric = gen_metric(outputs, labels, self._device, loss)
            metric_list.add_metric(metric)
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_metric = metric_list.squeeze()
        print(f'train cost time: {time.time() - start_time}')
        return train_metric

    def predict(self, *args, **kwargs):
        pass

    def evaluate(self, data_loader: GeneralDataLoader, *args, **kwargs) -> Metric:
        start_time = time.time()
        self._net.eval()
        metric_list = MetricSequence()
        with torch.no_grad():
            for val_data, val_labels in data_loader:
                inputs = val_data.to(self._device)
                labels = val_labels.to(self._device)
                outputs, loss = self._net(inputs)
                metric = gen_metric(outputs, labels, self._device, loss)
                metric_list.add_metric(metric)
        train_metric = metric_list.squeeze()
        print(f'train cost time: {time.time() - start_time}')
        return train_metric

    def save(self, path: str='.', model_name: str='model'):
        model_path = os.path.join(path, f'{model_name}.pth')
        torch.save(self._net.state_dict(), model_path)

    def load(self, path: str='.', model_name: str='model'):
        model_path = os.path.join(path, f'{model_name}.pth')
        self._net.load_state_dict(torch.load(model_path))
