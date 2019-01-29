# cython: language_level=3

# noinspection PyUnresolvedReferences
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport array

cdef struct Node:
    Node*left
    Node*right
    Node*child
    unsigned char key
    unsigned long*value

cdef struct Counter:
    unsigned long num_unique_keys
    unsigned long num_total_keys

cdef Node*create_node():
    node = <Node*> PyMem_Malloc(sizeof(Node))
    node.left = NULL
    node.right = NULL
    node.child = NULL
    node.value = NULL
    node.key = 0
    return node

cdef void free_post_order(Node*node):
    if node:
        free_post_order(node.child)
        free_post_order(node.left)
        free_post_order(node.right)
        PyMem_Free(node.value)
        PyMem_Free(node.child)
        PyMem_Free(node.left)
        PyMem_Free(node.right)

cdef void get_memory_size(Node*node, unsigned long*num_bytes):
    if node:
        get_memory_size(node.child, num_bytes)
        get_memory_size(node.left, num_bytes)
        get_memory_size(node.right, num_bytes)
        if node.value:
            num_bytes[0] += sizeof(node.value)
        num_bytes[0] += sizeof(node)

cdef class IndexCore:
    cdef Node root_node
    cdef unsigned short alloc_size_per_time
    cdef Counter cnt
    cdef unsigned short bytes_per_vector
    cdef unsigned char*_chunk_data

    cdef unsigned long get_memory_size(self):
        cdef unsigned long mem_usg
        mem_usg = 0
        get_memory_size(&self.root_node, &mem_usg)
        return mem_usg

    cdef void _index_value(self, Node*node):
        if node.value and node.value[0] == node.value[1]:
            new_value = <unsigned long*> PyMem_Realloc(node.value,
                                                       (node.value[
                                                            1] + self.alloc_size_per_time) * sizeof(unsigned long))
            if not new_value:
                raise MemoryError()
            node.value = new_value
            node.value[1] += self.alloc_size_per_time
        elif not node.value:
            node.value = <unsigned long*> PyMem_Malloc(self.alloc_size_per_time * sizeof(unsigned long))
            if not node.value:
                raise MemoryError()
            node.value[0] = 0
            node.value[1] = self.alloc_size_per_time - 2  # first two are reserved for counting
            self.cnt.num_unique_keys += 1
        (node.value + node.value[0] + 2)[0] = self.cnt.num_total_keys
        node.value[0] += 1
        self.cnt.num_total_keys += 1

    cpdef void index_chunk(self, unsigned char *data, const unsigned long num_total):
        self._chunk_data = data
        self.cnt.num_total_keys = num_total

    cpdef find_all_chunk(self, unsigned char *query):
        cdef array.array final_result = array.array('L')
        cdef unsigned char *pt
        cdef unsigned long _0
        cdef unsigned short _1
        cdef unsigned char is_match
        pt = self._chunk_data
        for _0 in range(self.cnt.num_total_keys):
            is_match = 1
            for _1 in range(self.bytes_per_vector):
                if (query + _1)[0] != (pt + _1)[0]:
                    is_match = 0
                    break
            if is_match == 1:
                final_result.append(_0)
            pt += self.bytes_per_vector
        return final_result

    cpdef contains_chunk(self, unsigned char *query, const unsigned long num_query):
        cdef array.array final_result = array.array('B', [0] * num_query)
        cdef unsigned char *pt
        cdef unsigned char *q_pt
        cdef unsigned long _0
        cdef unsigned long _1
        cdef unsigned short _2
        cdef unsigned char is_match
        pt = self._chunk_data
        for _0 in range(self.cnt.num_total_keys):
            q_pt = query
            for _1 in range(num_query):
                is_match = 1
                for _2 in range(self.bytes_per_vector):
                    if (q_pt + _2)[0] != (pt + _2)[0]:
                        is_match = 0
                        break
                if is_match == 1:
                    final_result[_1] = 1
                q_pt += self.bytes_per_vector
            pt += self.bytes_per_vector
        return final_result

    cpdef find_all_trie(self, unsigned char *query):
        cdef array.array final_result = array.array('L')
        cdef Node*node
        cdef unsigned short _1
        cdef unsigned long _2
        cdef unsigned char is_match
        node = &self.root_node
        is_match = 1
        for _1 in range(self.bytes_per_vector):
            key = query[_1]
            while node:
                if node.key == key:
                    if not node.child:
                        is_match = 0
                        break
                    else:
                        node = node.child
                        break
                elif key < node.key:
                    if not node.left:
                        is_match = 0
                        break
                    else:
                        node = node.left
                elif key > node.key:
                    if not node.right:
                        is_match = 0
                        break
                    else:
                        node = node.right
            if not node:
                is_match = 0
            if is_match == 0:
                break
        if is_match == 1 and node.value:
            for _2 in range(2, node.value[0] + 2):
                final_result.append(node.value[_2])
        return final_result

    cpdef contains_trie(self, unsigned char *query, const unsigned long num_query):
        cdef array.array final_result = array.array('B', [0] * num_query)
        cdef Node*node
        cdef unsigned char *q_pt
        cdef unsigned long _0
        cdef unsigned short _1
        cdef unsigned char is_match
        q_pt = query
        for _0 in range(num_query):
            node = &self.root_node
            is_match = 1
            for _1 in range(self.bytes_per_vector):
                key = q_pt[_1]
                while node:
                    if node.key == key:
                        if not node.child:
                            is_match = 0
                            break
                        else:
                            node = node.child
                            break
                    elif key < node.key:
                        if not node.left:
                            is_match = 0
                            break
                        else:
                            node = node.left
                    elif key > node.key:
                        if not node.right:
                            is_match = 0
                            break
                        else:
                            node = node.right
                if not node:
                    is_match = 0
                if is_match == 0:
                    break
            if is_match == 1 and node.value:
                final_result[_0] = 1
            q_pt += self.bytes_per_vector
        return final_result

    cpdef void index_trie(self, unsigned char *data, const unsigned long num_total):
        cdef Node*node
        cdef unsigned long _0
        cdef unsigned short _1
        for _0 in range(num_total):
            node = &self.root_node
            for _1 in range(self.bytes_per_vector):
                key = data[_1]
                while node:
                    if node.key == 0 or node.key == key:
                        node.key = key
                        if not node.child:
                            node.child = create_node()
                        node = node.child
                        break
                    elif key < node.key:
                        if not node.left:
                            node.left = create_node()
                        node = node.left
                    elif key > node.key:
                        if not node.right:
                            node.right = create_node()
                        node = node.right
            self._index_value(node)
            data += self.bytes_per_vector

    @property
    def counter(self):
        return {
            'num_keys': self.cnt.num_total_keys,
            'num_unique_keys': self.cnt.num_unique_keys
        }

    @property
    def size(self):
        return self.cnt.num_total_keys

    @property
    def memory_size(self):
        return self.get_memory_size()

    def __cinit__(self, bytes_per_vector):
        self.alloc_size_per_time = 100
        self.bytes_per_vector = bytes_per_vector

    def __dealloc__(self):
        free_post_order(&self.root_node)
