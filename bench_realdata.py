import sys
import time

import numpy as np
from hnsw_cpy import HnswIndex
from hnsw_cpy.cython_core.heappq import PriorityQueue
from hnsw_cpy.cython_core.hnsw import hamming_dist

from tqdm import tqdm
from collections import defaultdict


def toy_data_generator(size, bytes_num):
    return np.random.randint(0, 255, [size, bytes_num], dtype=np.uint8)

def from_file(filename, bytes_num):
    bin_data = open(filename, 'rb').read()
    num_total = int(len(bin_data) / bytes_num)
    vectors = np.frombuffer(bin_data, dtype=np.uint8).reshape(
            num_total, bytes_num)
    return vectors

def bt_query(vectors, ids, keys, top_k: int, bytes_num: int):
    keys = keys.reshape([-1, 1, bytes_num])
    dist = keys - np.expand_dims(vectors, axis=0)
    dist = np.sum(np.minimum(np.abs(dist), 1), -1)
    ret = []

    for _ids in dist:
        rk = sorted(enumerate(_ids), key=lambda x: x[1])
        ret.append([(ids[rk[i][0]].tolist(), rk[i][1]) for i in range(top_k)])

    return ret


if __name__ == '__main__':
    bytes_num = 20
    #max_iter = 5
    #num_repeat = 3
    top_k = 10
    # data_size = 10000
    query_size = 250
    total_size = 0

    print(f'Benchmarking for HNSW indexer')
    print('Data size\tIndex QPS\tIndex time(s)\tQuery QPS\tQuery times(s)')

    build_time_cost = []
    query_time_cost = []

    hnsw = HnswIndex(bytes_num, m=15, ef_construction=150)
    doc_vectors = from_file(sys.argv[1], bytes_num)
    data_size = doc_vectors.shape[0]
    doc_ids = np.array(list(range(data_size)))

    build_start_t = time.perf_counter()
    for i in tqdm(range(data_size)):
        vb = doc_vectors[i,:].tobytes()
        hnsw.index(i, vb)

    build_time_cost.append(time.perf_counter() - build_start_t)


    bt_result = bt_query(doc_vectors, doc_ids, doc_vectors[-query_size:, :], top_k, bytes_num)

    _query_time_cost = []
    query_start_t = time.perf_counter()
    recalls = []
    for qid in range(query_size):
        # query_data = np.random.randint(1, 255, bytes_num, dtype=np.uint8).tobytes()
        query_data = doc_vectors[data_size-query_size+qid,:].tobytes()

        h_r = [(r['id'], int(r['distance'])) for r in hnsw.query(query_data, top_k)]
        b_r = bt_result[qid]

        # h_count = defaultdict(int)
        b_count = defaultdict(int)
        max_count = 0
        for r in b_r:
            d = int(r[1])
            if d > max_count:
                max_count = d
            b_count[d] += 1

        c = 0
        m_count = 0
        for r in h_r:
            d = int(r[1])
            if d == max_count:
                m_count += 1
                continue

            if b_count.get(d, 0) > 0:
                c += 1
                b_count[d] -= 1

        c += m_count

        print(h_r)
        print(b_r)

        print(c)
        recalls.append(c / top_k)
        print()

        _query_time_cost.append(time.perf_counter() - query_start_t)
    query_time_cost.append(np.mean(_query_time_cost))

    build_t_avg = np.mean(build_time_cost)
    query_t_avg = np.mean(query_time_cost)
    build_qps = data_size / build_t_avg
    query_qps = query_size / query_t_avg
    recall = np.mean(recalls)
    print(f'{total_size}\t{build_qps}\t{build_t_avg}\t{query_qps}\t{query_t_avg}\t{recall}')
