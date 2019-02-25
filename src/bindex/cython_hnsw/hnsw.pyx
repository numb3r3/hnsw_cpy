# cython: language_level=3, wraparound=False, boundscheck=False

# noinspection PyUnresolvedReferences
#from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cymem.cymem cimport Pool
from cpython cimport array
from libc.stdlib cimport malloc, free
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.list cimport list as cpplist

from ctypes import c_ushort

import numpy as np

ctypedef unsigned int UIDX
DEF alloc_size_per_time = 200

cdef struct Counter:
    UIDX num_total

ctypedef cpp_map[UIDX, float]hnswEdgeSet

cdef struct hnswNode:
     UIDX id
     unsigned char*key
     unsigned short level
     hnswEdgeSet*edges




cdef hnswNode*create_node(UIDX id, unsigned short level, unsigned char *key):
    cdef hnswNode *node = <hnswNode*> malloc(sizeof(hnswNode))
    node.id = id
    node.level = level
    node.key = key
    return node

cdef class IndexHnsw:
    cdef Pool mem
    cdef Counter cnt
    cdef unsigned short bytes_per_vector
    cdef unsigned short max_level
    cdef cpp_vector[hnswNode*] nodes
    cdef hnswNode* entry_ptr

    cpdef void index(self, unsigned char *data, const UIDX num_total):
        cdef UIDX _0
        cdef unsigned short _1
        cdef unsigned char *key

        for _0 in range(num_total):
            key = <unsigned char*>self.mem.alloc(self.bytes_per_vector, sizeof(char))
            for _1 in range(self.bytes_per_vector):
                key[_1] = data[_1]
            node = create_node(self.size, c_ushort(0), key)
            self._add_node(node)
            self.nodes.push_back(node)
            data += self.bytes_per_vector

        self.cnt.num_total += num_total

    cdef _add_node(self, hnswNode *node):
        cdef unsigned short level = self._random_level()

    cdef unsigned short _random_level(self):
        #return c_ushort(0)
        return np.random.exponential(10)

    cpdef query(self, unsigned char *query):
        return None

    cpdef batch_query(self, unsigned char *query, const UIDX num_query):
        return None

    cpdef void save_model(self, model_path):
        pass

    cpdef void load_model(self, model_path):
        pass

    @property
    def size(self):
        return self.cnt.num_total

    @property
    def memory_size(self):
        raise NotImplemented
        #return get_memory_size(self.root_node)

    def __cinit__(self, bytes_per_vector):
        self.bytes_per_vector = bytes_per_vector
        self.mem = Pool()

    def __dealloc__(self):
        # self.free_trie()
        pass

    cpdef destroy(self):
        pass
