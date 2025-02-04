# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/4 11:33
import unittest

from src.language_model.bi_gram_nn import BiGramNN


class BiGramNNTestCase(unittest.TestCase):
    def setUp(self):
        self._corpus_path = 'test/test_language_model/data/names.txt'
        with open(self._corpus_path) as corpus_file:
            self._corpus = corpus_file.read().split('\n')
        self._bi_gram_nn = BiGramNN(class_dim=27)

    def test_bi_gram_nn(self):
        train_set = []
        train_label = []
        for line in self._corpus:
            padded_line = '.' + line + '.'
            for char1, char2 in zip(padded_line, padded_line[1:]):
                train_set.append(self._char_to_i(char1))
                train_label.append(self._char_to_i(char2))
        self._bi_gram_nn.train(train_set, train_label, epochs=100, learning_rate=50)
        test_str = 'andrejq'
        loss = self._bi_gram_nn.predict(test_str)
        print(loss)


    def tearDown(self):
        pass

    def _char_to_i(self, char):
        if char != '.':
            return ord(char) - 97
        else:
            return 26

if __name__ == '__main__':
    unittest.main()
