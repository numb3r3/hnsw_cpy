from bt_index import BTIndex
from bt_index.helper import get_run_args

if __name__ == '__main__':
    args = get_run_args()
    bt = BTIndex(args.binary_file.read(), args.bytes_per_vector)
    print(vars(bt.stat))
    print(bt.get_all_list_len())
