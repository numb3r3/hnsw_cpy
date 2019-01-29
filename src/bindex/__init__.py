from bindex.cython_core import IndexCore

from bindex.helper import set_logger

__version__ = '0.0.1'


class BIndex:
    def __init__(self, bytes_per_vector, index_mode='trie', verbose=False):
        self.logger = set_logger('BIndex', verbose)
        self.bytes_per_vector = bytes_per_vector
        self.indexer = IndexCore(bytes_per_vector)
        self.index_mode = index_mode
        if index_mode == 'trie':
            self._index_fn = lambda x: self.indexer.index_trie(x, self._get_size(x))
            self._contains_fn = lambda x: self.indexer.contains_trie(x, self._get_size(x))
            self._find_fn_single = lambda x: self.indexer.find_trie(x)
            self._find_fn_batch = lambda x, y: self.indexer.find_batch_trie(x, y)
        elif index_mode == 'none':
            self._index_fn = lambda x: self.indexer.index_chunk(x, self._get_size(x))
            self._contains_fn = lambda x: self.indexer.contains_chunk(x, self._get_size(x))
            self._find_fn_single = lambda x: self.indexer.find_chunk(x)
            self._find_fn_batch = lambda x, y: self.indexer.find_batch_chunk(x, y)
        else:
            raise NotImplementedError(f'mode="{index_mode}" is not implemented!')

    def _get_size(self, data):
        num_total = int(len(data) / self.bytes_per_vector)
        self.logger.info(f'input size: {len(data)} bytes; '
                         f'bytes/vector: {self.bytes_per_vector}; '
                         f'num vectors: {num_total}; '
                         f'mode: {self.index_mode}')
        return num_total

    def add(self, data):
        self._index_fn(data)

    def contains(self, query):
        result = self._contains_fn(query)
        return [v > 0 for v in result]

    def find(self, query):
        num_query = self._get_size(query)
        if num_query == 1:
            return self._find_fn_single(query).tolist()
        else:
            q_idx, d_idx = self._find_fn_batch(query, num_query)
            result = [[] for _ in range(num_query)]
            for (q, d) in zip(q_idx, d_idx):
                result[q].append(d)
            return result

    @property
    def statistic(self):
        return self.indexer.counter

    @property
    def size(self):
        return self.indexer.size

    @property
    def memory_size(self):
        return self.indexer.memory_size
