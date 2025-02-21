# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/20 17:18
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Tokenizer

from src.nano_GPT.config.model_config import ModelConfig
from src.nano_GPT.config.nano_gpt_data_set import NanoGPTDataset
from src.nano_GPT.model.nano_gpt_model import NanoGPTModel
from src.nano_GPT.utils import plot_loss


class NanoGPTTestCase(unittest.TestCase):
    def setUp(self):
        self._data_path = './data/20'
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._batch_size = 64
        self._epochs = 10
        self._interval = 1
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._max_lines = 100
        self._model_config = ModelConfig()
        self._model = NanoGPTModel(self._model_config)

    def test_nano_gpt(self):
        total_params = sum(p.numel() for p in self._model.parameters())
        print(f"Total parameters: {total_params / 1e6} M")
        train_data_loader, val_data_loader = self._load_train_data()
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=3e-4)
        # 设置 cosine 学习率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        train_loss_list = []
        val_loss_list = []
        max_val_f1 = 0
        for epoch in range(self._epochs):
            train_metric = self._model.train(train_data_loader, optimizer, scheduler)
            print(f'epoch: {epoch}, train_metric: {train_metric}')
            if (epoch + 1) % self._interval == 0:
                val_metric = self._model.evaluate(val_data_loader)
                val_loss_list.append(val_metric.loss)
                train_loss_list.append(train_metric.loss)
                print(f'val metric: {val_metric}')
                if val_metric.f1_score > max_val_f1:
                    print(f'new max_val_f1_score: {val_metric.f1_score} in epoch: {epoch}, save model')
                    max_val_f1 = val_metric.f1_score
                    self._model.save(path='data', model_name='stock_cnn_best_f1.pth')
        plot_loss(train_loss_list, val_loss_list)
        self.assertEqual(True, False)

    def tearDown(self):
        pass

    def _load_train_data(self):
        labels_df = pd.read_csv('data/label_file.csv')
        dataset = NanoGPTDataset(data_path=self._data_path, seq_len=self._model_config.seq_len,
                                 tokenizer=self._tokenizer, max_lines=self._max_lines)
        val_split = 0.2
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader


if __name__ == '__main__':
    unittest.main()
