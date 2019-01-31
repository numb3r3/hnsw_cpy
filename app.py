from bindex import BIndex
from bindex.helper import get_run_args

if __name__ == '__main__':
    args = get_run_args()
    bt = BIndex(args.bytes_per_vector)
    bt.add(args.binary_file.read(), num_rows=args.num_vector)
    print(bt.statistic)
