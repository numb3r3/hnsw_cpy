from typing import List, Tuple
import random
import unittest

import numpy as np
from hnsw_cpy import HnswIndex
from hnsw_cpy.cython_hnsw.hnsw import hamming_dist


class TestHnswIndex(unittest.TestCase):
    def setUp(self):
        self.toy_data = {
            'data': [bytes([1, 2, 3, 4]),
                     bytes([1, 2, 3, 5]),
                     bytes([1, 2, 3, 2]),
                     bytes([1, 2, 3, 4]),
                     bytes([5, 1, 2, 3]),
                     bytes([5, 1, 2, 4]),
                     bytes([5, 1, 2, 3]),
                     bytes([1, 1, 1, 1])],
            'query': [bytes([1, 2, 3 ,4]),
                      bytes([5, 1, 2, 3]),
                      bytes([1, 1, 1, 1])],
            'bytes': 4,
            'expect': [[0, 3], [4, 6], [7]]
        }

        self.hnsw_toy = HnswIndex(self.toy_data['bytes'])

        self.hnsw_toy.bulk_index(self.toy_data['data'], range(8))


    def test_hamming(self):
        x = bytes([1, 2, 3, 4])
        y = bytes([1, 2, 3, 5])
        z = bytes([2, 3, 5, 1])
        dist0 = hamming_dist(x, x)
        dist1 = hamming_dist(x, y)
        dist2 = hamming_dist(x, z)
        self.assertEqual(dist0, 0)
        self.assertEqual(dist1, 1)
        self.assertEqual(dist2, 7)


    def test_add_data(self):
        self.assertEqual(self.hnsw_toy.bytes_per_vector, self.toy_data['bytes'])
        self.assertEqual(self.hnsw_toy.size, 8)

    def test_query(self):
        for query, expect in zip(self.toy_data['query'], self.toy_data['expect']):
            result = self.hnsw_toy.query(query, 10)
            for i in range(len(expect)):
                self.assertEqual(result[i]['distance'], 0)


if __name__ == '__main__':
    unittest.main()
