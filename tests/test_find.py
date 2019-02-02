import random
import unittest

import numpy as np
from bindex import BIndex


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.toy1 = {'data': bytes([1, 2, 3, 4,
                                    1, 2, 3, 5,
                                    1, 2, 3, 2,
                                    1, 2, 3, 4,
                                    5, 1, 2, 3,
                                    5, 1, 2, 4,
                                    5, 1, 2, 3,
                                    1, 1, 1, 1]),
                     'query': bytes([1, 2, 3, 4]),
                     'bytes': 4,
                     'expect': [0, 3]}

        self.toy2 = {'data': bytes([1, 2, 3, 4,
                                    1, 2, 3, 5,
                                    1, 2, 3, 2,
                                    1, 2, 3, 4,
                                    5, 1, 2, 3,
                                    5, 1, 2, 4,
                                    5, 1, 2, 3,
                                    1, 1, 1, 1]),
                     'query': bytes([5, 6, 7, 8]),
                     'bytes': 4,
                     'expect': []}

        tmp = np.random.randint(1, 255, [10000, 512], dtype=np.uint8)
        query = tmp[random.randint(0, 10000)]
        result = (tmp == query).all(axis=1).nonzero()[0].tolist()

        self.toy3 = {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'query': query.tobytes(),
            'expect': result
        }

    def _test_toy_data(self, toydata, store_mode):
        bt = BIndex(toydata['bytes'], store_mode)
        bt.add(toydata['data'])
        self.assertEqual(toydata['expect'], bt.find(toydata['query']))

    def test_basic(self):
        self._test_toy_data(self.toy1, 'none')

    def test_trie(self):
        self._test_toy_data(self.toy1, 'trie')

    def test_basic_empty(self):
        self._test_toy_data(self.toy2, 'none')

    def test_trie_empty(self):
        self._test_toy_data(self.toy2, 'trie')

    def test_basic_large(self):
        self._test_toy_data(self.toy3, 'none')

    def test_trie_large(self):
        self._test_toy_data(self.toy3, 'trie')


if __name__ == '__main__':
    unittest.main()
