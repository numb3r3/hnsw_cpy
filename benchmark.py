import time

from bindex import BIndex
from bindex.helper import get_run_args

if __name__ == '__main__':
    args = get_run_args()
    num_vector = 16384
    print('data size\tQPS\ttime(s)\tmemory\tunique keys\tunique rate')
    bt = BIndex(args.bytes_per_vector)
    bin_data = args.binary_file.read()
    for _ in range(10):
        start_t = time.perf_counter()
        bt.add(bin_data, num_rows=num_vector)
        t_avg = time.perf_counter() - start_t
        m_avg = bt.memory_size
        unique_rate = int(100 * bt.statistic['num_unique_keys'] / bt.statistic['num_keys'])
        qps = int(num_vector / t_avg)
        print(f'{num_vector}\t{qps}\t{t_avg}\t{m_avg}\t{bt.statistic["num_unique_keys"]}\t{unique_rate}')
        num_vector *= 2
        bt.clear()