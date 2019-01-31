import time

from bindex import BIndex
from bindex.helper import get_run_args

if __name__ == '__main__':
    args = get_run_args()

    num_vector = 8192
    print('data size\tQPS\ttime(s)\tmemory\tunique keys\tunique rate')
    for _ in range(10):
        bt = BIndex(args.bytes_per_vector)
        start_t = time.perf_counter()
        bt.add(args.binary_file.read(), num_rows=num_vector)
        time_cost = time.perf_counter() - start_t
        mem_cost = bt.memory_size
        unique_rate = int(100 * bt.statistic['num_unique_keys'] / bt.statistic['num_keys'])
        qps = int(num_vector / time_cost)
        print(f'{num_vector}\t{qps}\t{time_cost}\t{m_avg}\t{bt.statistic["num_unique_keys"]}\t{unique_rate}')
        num_vector *= 2
