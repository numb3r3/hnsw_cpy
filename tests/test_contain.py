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
                     'query': bytes([1, 2, 3, 5,
                                     6, 1, 2, 3,
                                     5, 1, 2, 3,
                                     1, 2, 3, 4,
                                     1, 2, 3, 5,
                                     6, 1, 2, 3]),
                     'bytes': 4,
                     'expect': [True, False, True, True, True, False]}

        tmp = np.random.randint(1, 255, [10000, 512], dtype=np.uint8)
        tmp1 = tmp.copy()
        np.random.shuffle(tmp1)
        self.toy2 = {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'query': tmp1.tobytes(),
            'expect': [True] * tmp.shape[0]
        }

        tmp2 = np.random.randint(1, 128, [10000, 512], dtype=np.uint8)
        tmp3 = np.random.randint(128, 255, [10000, 512], dtype=np.uint8)
        self.toy3 = {
            'data': tmp2.tobytes(),
            'bytes': tmp2.shape[1],
            'query': tmp3.tobytes(),
            'expect': [False] * tmp.shape[0]
        }

    def _test_toy_data(self, toydata, store_mode):
        bt = BIndex(toydata['bytes'], store_mode)
        bt.add(toydata['data'])
        self.assertEqual(toydata['expect'], bt.contains(toydata['query']))

    def test_basic(self):
        self._test_toy_data(self.toy1, 'none')

    def test_trie(self):
        self._test_toy_data(self.toy1, 'trie')

    def test_basic_large(self):
        self._test_toy_data(self.toy2, 'none')

    def test_trie_large(self):
        self._test_toy_data(self.toy2, 'trie')

    def test_basic_negative(self):
        self._test_toy_data(self.toy3, 'none')

    def test_trie_negative(self):
        self._test_toy_data(self.toy3, 'trie')


if __name__ == '__main__':
    unittest.main()
