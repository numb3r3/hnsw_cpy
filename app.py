from bt_index.indexer import BTIndex

if __name__ == '__main__':
    with open('test.bin', 'rb') as fp:
        bt = BTIndex(fp.read(), bytes_per_vector=96)
    print(vars(bt.stat))
    print(bt.get_all_list_len())