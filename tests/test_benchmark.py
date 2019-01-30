import time
import unittest

import numpy as np
from bindex import BIndex


class TestIndexer(unittest.TestCase):
    @staticmethod
    def build_toy_data(size, dim):
        tmp = np.random.randint(1, 255, [size, dim], dtype=np.uint8)
        return {
            'data': tmp.tobytes(),
            'bytes': tmp.shape[1],
            'num_keys': tmp.shape[0],
            'num_unique_keys': np.unique(tmp, axis=0).shape[0]
        }

    def test_index_trie(self):
        self._benchmark('trie', 'index')

    def test_find_batch(self):
        self._benchmark('trie', 'find')
        self._benchmark('none', 'find')

    def _benchmark(self, index_mode, benchmark_fn, num_repeat=10):
        print(f'\nbenchmarking {benchmark_fn} for mode {index_mode} (avg. over {num_repeat})')
        print('data size\tQPS\ttime(s)\tmemory')
        data_size = 256
        query_size = 512
        data_dim = 96
        time_cost = []
        mem_size = []
        for j in range(10):
            time_cost.clear()
            mem_size.clear()
            for _ in range(num_repeat):
                toy_data = self.build_toy_data(data_size, data_dim)
                bt = BIndex(bytes_per_vector=toy_data['bytes'], index_mode=index_mode)
                start_t = time.perf_counter()
                bt.add(toy_data['data'])
                if benchmark_fn == 'find':
                    query_data = self.build_toy_data(query_size, data_dim)['data']
                    start_t = time.perf_counter()
                    bt.find(query_data)
                time_cost.append(time.perf_counter() - start_t)
                mem_size.append(bt.memory_size)
                bt.destroy()
            t_avg = np.mean(time_cost)
            m_avg = int(np.mean(mem_size))
            qps = int((query_size if benchmark_fn == 'find' else data_size) / t_avg)
            print(f'{data_size}\t{qps}\t{t_avg}\t{m_avg}')
            data_size *= 2


if __name__ == '__main__':
    unittest.main()
