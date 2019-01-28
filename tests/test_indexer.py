import unittest

import numpy as np
from bindex import BIndex


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.toy_data1 = {'data': bytes([1, 2, 3, 4,
                                         1, 2, 3, 5,
                                         1, 2, 3, 2,
                                         1, 2, 3, 4,
                                         5, 1, 2, 3,
                                         5, 1, 2, 4,
                                         5, 1, 2, 3,
                                         1, 1, 1, 1]),
                          'bytes': 4,
                          'num_keys': 8,
                          'num_unique_keys': 6}

        tmp = np.random.randint(0, 255, [10000, 512], dtype=np.uint8)
        self.toy_data2 = {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'num_keys': tmp.shape[0],
            'num_unique_keys': np.unique(tmp, axis=0).shape[0]
        }

    def _test_toy_data(self, toy_data):
        bt = BIndex(bytes_per_vector=toy_data['bytes'])
        bt.add(toy_data['data'])
        print(f'size of the tree: {bt.memory_size} bytes')
        self.assertEqual(bt.statistic['num_keys'], toy_data['num_keys'],
                         'number of total keys is not correct!')
        self.assertEqual(bt.statistic['num_unique_keys'], toy_data['num_unique_keys'],
                         'number of unique keys is not correct!')

    def test_basic(self):
        self._test_toy_data(self.toy_data1)

    def test_random(self):
        self._test_toy_data(self.toy_data2)


if __name__ == '__main__':
    unittest.main()
