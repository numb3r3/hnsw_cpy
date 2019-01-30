import ctypes

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
            self._index_fn = self.indexer.index_trie
            self._contains_fn = self.indexer.contains_trie
            self._find_fn_single = self.indexer.find_trie
            self._find_fn_batch = self.indexer.find_batch_trie
        elif index_mode == 'none':
            self._index_fn = self.indexer.index_chunk
            self._contains_fn = self.indexer.contains_chunk
            self._find_fn_single = self.indexer.find_chunk
            self._find_fn_batch = self.indexer.find_batch_chunk
        else:
            raise NotImplementedError(f'mode="{index_mode}" is not implemented!')

    def _get_size(self, data):
        if not isinstance(data, bytes):
            raise TypeError(f'"data" must be "bytes", received {type(data)}')
        len_data = len(data)
        if len_data < self.bytes_per_vector:
            raise IndexError(
                f'"data" is {len_data} bytes, shorter than predefined "bytes_per_vector={self.bytes_per_vector}"')
        if len_data % self.bytes_per_vector != 0:
            raise ValueError(
                f'"data" is {len_data} bytes,'
                f'can not be divided by predefined "bytes_per_vector={self.bytes_per_vector}"')
        num_total = int(len(data) / self.bytes_per_vector)
        self.logger.debug(f'input size: {len(data)} bytes; '
                          f'bytes/vector: {self.bytes_per_vector}; '
                          f'num vectors: {num_total}; '
                          f'mode: {self.index_mode}')
        if num_total > ctypes.c_uint(-1).value:
            raise IndexError(
                f'"data" contains {num_total} rows, larger than the upper bound {ctypes.c_uint(-1).value}!')
        return num_total

    def add(self, data, num_rows=None):
        """ Add data to the index

        Multiple `add()` behave like appending. For example:
        .. highlight:: python
        .. code-block:: python

            bt = BIndex(4)
            bt.add(bytes([1, 2, 3, 4,
                          5, 6, 7, 8]))
            bt.add(bytes([5, 6, 7, 8]))
            print(bt.size)  # yields 3

        :type data: bytes
        :type num_rows: int
        :param data: data to be indexed in bytes
        :param num_rows: number of rows to index
        :rtype: List[bool]
        """
        if not num_rows:
            num_rows = self._get_size(data)
        self._index_fn(data, num_rows)

    def contains(self, data, num_rows=None):
        """ Check if the index contains data.

        :type data: bytes
        :type num_rows: int
        :param data: query data in bytes
        :param num_rows: number of rows
        :rtype: List[bool]
        """
        if not num_rows:
            num_rows = self._get_size(data)
        result = self._contains_fn(data, num_rows)
        return [v > 0 for v in result]

    def find(self, query, num_rows=None):
        if not num_rows:
            num_rows = self._get_size(query)

        if num_rows == 1:
            return self._find_fn_single(query).tolist()
        else:
            q_idx, d_idx = self._find_fn_batch(query, num_rows)
            result = [[] for _ in range(num_rows)]
            for (q, d) in zip(q_idx, d_idx):
                result[q].append(d)
            return result

    @property
    def statistic(self):
        return self.indexer.counter

    @property
    def shape(self):
        """
        :rtype: Tuple[int, int]
        :return: a tuple of (number of elements, number of bytes per vector) of the index
        """
        return self.indexer.size, self.bytes_per_vector

    @property
    def memory_size(self):
        """
        :return: estimated memory size of the index, only valid when "index_mode='trie'"
        """
        return self.indexer.memory_size

    def clear(self):
        """ remove all elements from the index
        """
        self.indexer.destroy()
