# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/3 20:40
import unittest

from src.language_model.bi_gram import BiGram


class BiGramTestCase(unittest.TestCase):
    def setUp(self):
        self._corpus_path = 'test/test_language_model/data/names.txt'
        with open(self._corpus_path) as corpus_file:
            self._corpus = corpus_file.read().split('\n')
        self._bi_gram = BiGram('.')

    def test_predict(self):
        self._test_str = 'andrejq'
        self._bi_gram.train(self._corpus)
        neg_log_likelihood = self._bi_gram.predict(self._test_str)
        print(f'negative log-likelihood of {self._test_str} is {neg_log_likelihood: .4f}')
        random_str = self._bi_gram.generate()
        print(f'random string is {random_str}')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
