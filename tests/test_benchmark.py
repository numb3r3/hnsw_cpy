import time

import unittest

import numpy as np
from hnsw_cpy import HnswIndex

class TestHnswIndex(unittest.TestCase):
    @staticmethod
    def toy_data_generator(size, dim):
        index = -1
        while index < size:
            tmp = np.random.randint(1, 255, dim, dtype=np.uint8)
            index += 1
            yield (index, tmp.tobytes())

    def test_index_hnsw(self):
        self._benchmark('index', self.toy_data_generator)

    def test_query_hnsw(self):
        self._benchmark('query', self.toy_data_generator)


    def _benchmark(self, benchmark_fn, data_gen_fn,
                   num_repeat=3, data_size=10000, data_dim=8, max_iter=5):
        print(f'\nbenchmarking {benchmark_fn} for HNSW index (avg. over {num_repeat})')
        print('data size\t|\tIndex QPS\t|\t Index time(s)\t|\tQuery QPS\t|\tQuery times(s)')

        query_size = 512
        build_time_cost = []
        query_time_cost = []
        # mem_size = []
        for j in range(max_iter):
            build_time_cost.clear()
            query_time_cost.clear()
            #mem_size.clear()
            for _ in range(num_repeat):
                hnsw = HnswIndex(bytes_per_vector=data_dim)
                build_start_t = time.perf_counter()
                for i, data in data_gen_fn(data_size, data_dim):
                    hnsw.index(i, data)
                build_time_cost.append(time.perf_counter() - build_start_t)

                _query_time_cost = []
                query_start_t = time.perf_counter()
                for qid in range(query_size):
                    query_data = np.random.randint(1, 255, data_dim, dtype=np.uint8).tobytes()
                    hnsw.query(query_data, 10)
                    _query_time_cost.append(time.perf_counter() - query_start_t)
                query_time_cost.append(np.mean(_query_time_cost))

                hnsw.clear()

            build_t_avg = np.mean(build_time_cost)
            query_t_avg = np.mean(query_time_cost)
            build_qps = data_size / build_t_avg
            query_qps = query_size / query_t_avg
            print(f'{data_size}\t|\t{build_qps}\t|\t{build_t_avg}\t|\t{query_qps}\t|\t{query_t_avg}')
            data_size *= 2


if __name__ == '__main__':
    unittest.main()
