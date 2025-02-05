# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/5 16:06
import unittest

from src.language_model.bi_gram_mlp import BiGramMLP


class BiGramMLPTestCase(unittest.TestCase):
    def setUp(self):
        self._corpus_path = 'test/test_language_model/data/names.txt'
        with open(self._corpus_path) as corpus_file:
            self._corpus = corpus_file.read().split('\n')
        self._bi_gram_mlp = BiGramMLP(embed_dim=3, hidden_dim=10, regular=0.001)

    def test_train(self):
        train_set = []
        train_label = []
        for line in self._corpus:
            padded_line = '.' + line + '.'
            for char1, char2 in zip(padded_line, padded_line[1:]):
                train_set.append(self._bi_gram_mlp._atoi[char1])
                train_label.append(self._bi_gram_mlp._atoi[char2])
        self._bi_gram_mlp.train(train_set, train_label, epochs=10000, learning_rate=.1)
        loss = self._bi_gram_mlp.predict('andrejq')
        print(loss)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
