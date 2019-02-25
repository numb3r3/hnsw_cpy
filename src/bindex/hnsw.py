import ctypes

from bindex.cython_hnsw.hnsw import IndexHnsw
from bindex.helper import set_logger

__version__ = '0.0.1'


class HnswIndex:
    def __init__(self, bytes_per_vector, verbose=False):
        """ A HNSW index object for storing and searching bytes vectors

        :type bytes_per_vector: int
        :type verbose: bool
        :param bytes_per_vector: number of bytes per vector
        :param verbose: enables extra logging
        """
        self.logger = set_logger('HnswIndex', verbose)
        self.bytes_per_vector = bytes_per_vector
        self.indexer = IndexHnsw(bytes_per_vector)
        self._index_fn = self.indexer.index
        self._find_fn_single = self.indexer.query
        self._find_fn_batch = self.indexer.batch_query
        self._save_model = self.indexer.save_model
        self._load_model = self.indexer.load_model



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
                          f'num vectors: {num_total}'
                          )

        if num_total > ctypes.c_uint(-1).value:
            raise IndexError(
                f'"data" contains {num_total} rows, larger than the upper bound {ctypes.c_uint(-1).value}!')
        return num_total

    def add(self, data, num_rows=None):
        """ Add data to the index

        Multiple `add()` behave like appending. For example:
        .. highlight:: python
        .. code-block:: python

            hix = HnswIndex(4)
            hix.add(bytes([1, 2, 3, 4,
                          5, 6, 7, 8]))
            hix.add(bytes([5, 6, 7, 8]))
            print(hix.size)  # yields 3

        :type data: bytes
        :type num_rows: int
        :param data: data to be indexed in bytes
        :param num_rows: number of rows to index
        :rtype: List[bool]
        """
        if not num_rows:
            num_rows = self._get_size(data)
        self._index_fn(data, num_rows)


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
    def size(self):
        return self.indexer.size

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

    def save_model(self, model_path):
        """save the index's model

        :type model_path: str
        :param model_path: the file path where the index's model will be dumped
        """
        self._save_model(model_path)

    def load_model(self, model_path):
        """load the index from the model file
        :type model_path: str
        :param model_path: the model's file path
        """
        self._load_model(model_path)

    def clear(self):
        """ remove all elements from the index
        """
        self.indexer.destroy()
