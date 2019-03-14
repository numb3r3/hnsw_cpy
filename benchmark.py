import time

import numpy as np
from hnsw_cpy import HnswIndex
from hnsw_cpy.cython_core.heappq import PriorityQueue
from hnsw_cpy.cython_core.hnsw import hamming_dist


def toy_data_generator(size, bytes_num):
    index = -1
    while index < size:
        tmp = np.random.randint(0, 255, bytes_num, dtype=np.uint8)
        index += 1
        yield (index, tmp.tobytes())

def brute_force_query(flat_vectors, query: bytes, top_k: int, bytes_num: int):
    pq = PriorityQueue()
    for i, vector in enumerate(flat_vectors):
        dist = hamming_dist(vector, query, bytes_num)
        pq.push(dist, (i, "x"))

    result = []
    for i in range(top_k):
        dist, item = pq.pop_min()
        result.append((item[0], dist))

    return result


if __name__ == '__main__':
    bytes_num = 20
    #max_iter = 5
    #num_repeat = 3
    data_size = 10000
    query_size = 125
    total_size = 0

    print(f'Benchmarking for HNSW indexer')
    print('Data size\tIndex QPS\tIndex time(s)\tQuery QPS\tQuery times(s)')

    build_time_cost = []
    query_time_cost = []

    hnsw = HnswIndex(bytes_num)
    flat_vectors = []

    while True:
        build_time_cost.clear()
        query_time_cost.clear()


        build_start_t = time.perf_counter()
        for i, data in toy_data_generator(data_size, bytes_num):
            hnsw.index(total_size+i, data)
            if total_size < query_size:
                flat_vectors.append(data)
        build_time_cost.append(time.perf_counter() - build_start_t)
        total_size += data_size

        _query_time_cost = []
        query_start_t = time.perf_counter()
        for qid in range(query_size):
            # query_data = np.random.randint(1, 255, bytes_num, dtype=np.uint8).tobytes()
            query_data = flat_vectors[qid]
            # f_r = [(r[0], r[1]) for r in brute_force_query(flat_vectors, query_data, 10, bytes_num)]
            # hnsw.query(query_data, 10)
            h_r = [(r['id'], int(r['distance'])) for r in hnsw.query(query_data, 10)]
            # print(f_r)
            print(h_r)
            _query_time_cost.append(time.perf_counter() - query_start_t)
        query_time_cost.append(np.mean(_query_time_cost))

        build_t_avg = np.mean(build_time_cost)
        query_t_avg = np.mean(query_time_cost)
        build_qps = data_size / build_t_avg
        query_qps = query_size / query_t_avg
        print(f'{total_size}\t{build_qps}\t{build_t_avg}\t{query_qps}\t{query_t_avg}')
        # break
