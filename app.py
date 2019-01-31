from bindex import BIndex
from bindex.helper import get_run_args

if __name__ == '__main__':
    args = get_run_args()
    bt = BIndex(args.bytes_per_vector)
    bt.add(args.binary_file.read(), num_rows=args.num_vector)
    print(bt.statistic)
    print(f'compression rate: %2.0f%%' % int(100 * bt.statistic['num_unique_keys'] / bt.statistic['num_total_keys']))
