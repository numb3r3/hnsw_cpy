from bindex import BIndex
from bindex.helper import get_run_args, TimeContext

if __name__ == '__main__':
    args = get_run_args()
    bt = BIndex(args.bytes_per_vector)
    with TimeContext('building index...'):
        bt.add(args.data_file.read(), args.num_data)

    with TimeContext('searching query...'):
        r = bt.find(args.query_file.read(), args.num_query)
        print(r)
