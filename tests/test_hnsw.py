import random
import unittest

import numpy as np
from bindex.hnsw import HnswIndex
from bindex.cython_hnsw.hnsw import hamming_dist


class TestHnswIndex(unittest.TestCase):
    def setUp(self):
        self.toy1 = {'data': bytes([1, 2, 3, 4,
                                    1, 2, 3, 5,
                                    1, 2, 3, 2,
                                    1, 2, 3, 4,
                                    5, 1, 2, 3,
                                    5, 1, 2, 4,
                                    5, 1, 2, 3,
                                    1, 1, 1, 1]),
                     'query': bytes([1, 2, 3, 4,
                                     5, 1, 2, 3,
                                     1, 1, 1, 1]),
                     'bytes': 4,
                     'expect': [[0, 3], [4, 6], [7]]}

        self.toy2 = {'data': bytes([1, 2, 3, 4,
                                    1, 2, 3, 5,
                                    1, 2, 3, 2,
                                    1, 2, 3, 4,
                                    5, 1, 2, 3,
                                    5, 1, 2, 4,
                                    5, 1, 2, 3,
                                    1, 1, 1, 1]),
                     'query': bytes([5, 6, 7, 8,
                                     5, 6, 7, 8,
                                     5, 7, 8, 8]),
                     'bytes': 4,
                     'expect': [[], [], []]}

        tmp = np.random.randint(1, 255, [10000, 512], dtype=np.uint8)
        query = tmp[random.randint(0, 10000)]
        result = (tmp == query).all(axis=1).nonzero()[0].tolist()

        self.toy3 = {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'query': query.tobytes(),
            'expect': result
        }

        self.hnsw_toy1 = HnswIndex(self.toy1['bytes'])

        self.hnsw_toy1.add(self.toy1['data'])


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
        self.assertEqual(self.hnsw_toy1.bytes_per_vector, self.toy1['bytes'])
        self.assertEqual(self.hnsw_toy1.size, 8)

    def test_query(self):
        result = self.hnsw_toy1.find(self.toy1['query'])
        print(result)


if __name__ == '__main__':
    unittest.main()
