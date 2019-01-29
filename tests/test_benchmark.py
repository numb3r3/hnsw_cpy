import time
import unittest

import numpy as np
from bindex import BIndex


class TestIndexer(unittest.TestCase):
    def build_toy_data(self, size, dim):
        tmp = np.random.randint(1, 255, [size, dim], dtype=np.uint8)
        return {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'num_keys': tmp.shape[0],
            'num_unique_keys': np.unique(tmp, axis=0).shape[0]
        }

    def _test_index_trie(self):
        data_size = 256
        data_dim = 256
        for j in range(10):
            toy_data = self.build_toy_data(data_size, data_dim)
            bt = BIndex(bytes_per_vector=toy_data['bytes'])
            start_t = time.perf_counter()
            bt.add(toy_data['data'])
            print(f'indexing {data_size}\t{time.perf_counter()-start_t}')
            data_size *= 2

    def test_find_batch(self):
        self.find_batch_mode('trie')
        self.find_batch_mode('none')

    def find_batch_mode(self, index_mode):
        print(f'benchmarking search for mode {index_mode}')
        data_size = 256
        query_size = 256
        data_dim = 256
        avg_time = []
        for j in range(10):
            for _ in range(5):
                toy_data = self.build_toy_data(data_size, data_dim)
                query_data = self.build_toy_data(query_size, data_dim)
                bt = BIndex(bytes_per_vector=toy_data['bytes'], index_mode=index_mode)
                bt.add(toy_data['data'])
                start_t = time.perf_counter()
                bt.find(query_data['data'])
                avg_time.append(time.perf_counter() - start_t)
            print(f'{data_size}\t{np.mean(avg_time)}')
            avg_time.clear()
            data_size *= 2


if __name__ == '__main__':
    unittest.main()
