from typing import List, Tuple
import ctypes

from hnsw_cpy.cython_core.hnsw import IndexHnsw
from hnsw_cpy.helper import set_logger

__version__ = '0.0.1'


class HnswIndex:
    """Hierarchical Navigable Small World (HNSW) data structure.

    Based on the work by Yury Malkov and Dmitry Yashunin, available at
    http://arxiv.org/pdf/1603.09320v2.pdf

    HNSWs allow performing approximate nearest neighbor search with
    arbitrary data and non-metric dissimilarity functions.
    """
    def __init__(self, bytes_num, verbose=False, **kwargs):
        """ A HNSW index object for storing and searching bytes vectors

        :type bytes_per_vector: int
        :param bytes_per_vector: number of bytes per vector
        """
        self.logger = set_logger('HnswIndex', verbose)
        self.bytes_num = bytes_num
        self.indexer = IndexHnsw(bytes_num)


    def _get_size(self, data):
        if not isinstance(data, bytes):
            raise TypeError(f'"data" must be "bytes", received {type(data)}')
        len_data = len(data)
        if len_data < self.bytes_num:
            raise IndexError(
                f'"data" is {len_data} bytes, shorter than predefined "bytes_per_vector={self.bytes_per_vector}"')
        if len_data % self.bytes_num != 0:
            raise ValueError(
                f'"data" is {len_data} bytes,'
                f'can not be divided by predefined "bytes_per_vector={self.bytes_per_vector}"')
        num_total = int(len(data) / self.bytes_num)
        self.logger.debug(f'input size: {len(data)} bytes; '
                          f'bytes/vector: {self.bytes_per_vector}; '
                          f'num vectors: {num_total}'
                          )

        if num_total > ctypes.c_uint(-1).value:
            raise IndexError(
                f'"data" contains {num_total} rows, larger than the upper bound {ctypes.c_uint(-1).value}!')
        return num_total

    def index(self, doc_id: int, vector: bytes):
        self.indexer.index(doc_id, vector)

    def bulk_index(self, doc_ids: List[int], vectors: List[bytes]):
        if len(vectors) != len(doc_ids):
            raise ValueError("the shape of vector list and doc list does not match")
        for vector, doc_id in zip(vectors, doc_ids):
            self.indexer.index(doc_id, vector)

    def query(self, query: bytes, top_k: int):
        return self.indexer.query(query, top_k)



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
