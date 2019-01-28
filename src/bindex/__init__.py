from bindex.cython_core import IndexCore

from bindex.helper import set_logger


class BIndex:
    def __init__(self, bytes_per_vector, verbose=False):
        self.logger = set_logger('BIndex', verbose)
        self.bytes_per_vector = bytes_per_vector
        self._index = IndexCore(bytes_per_vector)

    def add(self, data):
        num_vec = int(len(data) / self.bytes_per_vector)
        self.logger.info(f'input size: {len(data)} bytes;\t'
                         f'bytes/vector: {self.bytes_per_vector};\t'
                         f'expected number of vectors: {num_vec}')
        self._index.index_multi_vecs(data, num_vec)

    @property
    def statistic(self):
        return self._index.counter

    @property
    def size(self):
        return self._index.size

    @property
    def memory_size(self):
        return self._index.memory_size
