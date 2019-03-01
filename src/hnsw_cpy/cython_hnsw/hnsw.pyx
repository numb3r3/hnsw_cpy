# cython: language_level=3, wraparound=False, boundscheck=False

# noinspection PyUnresolvedReferences
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cymem.cymem cimport Pool
from cpython cimport array
from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, floor
from libcpp.map cimport map as cpp_map
from libcpp.set cimport set as cpp_set
from libcpp.vector cimport vector as cpp_vector
from libcpp.list cimport list as cpp_list


from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc

from ctypes import c_ushort

from hnsw_cpy.cython_hnsw.utils import PriorityQueue
from hnsw_cpy.cython_lib.prehash cimport prehash_map, prehash_insert, prehash_get


ctypedef unsigned int UIDX
DEF alloc_size_per_time = 200

cdef struct Counter:
    UIDX num_total

cdef struct hnsw_edge:
    UIDX node_id
    unsigned short dist
    hnsw_edge* next


cdef struct hnsw_edge_set:
    hnsw_edge* head_ptr
    hnsw_edge* last_ptr
    unsigned int size

cdef struct hnswNode:
     UIDX id
     unsigned char*key
     unsigned int level

     hnsw_edge_set**edges

cdef hnswNode* create_node(UIDX id, unsigned int level, unsigned char *key):
     cdef hnswNode *node = <hnswNode*> malloc(sizeof(hnswNode))
     node.id = id
     node.level = level
     node.key = key

     node.edges = <hnsw_edge_set**> malloc((level+1) * sizeof(hnsw_edge_set*))

     cdef hnsw_edge_set* edge_set
     for l in range(level+1):
         edge_set = <hnsw_edge_set*> malloc(sizeof(hnsw_edge_set))
         edge_set.size = 0
         edge_set.head_ptr = NULL
         edge_set.last_ptr = NULL

         node.edges[l] = edge_set

     return node

cdef void _add_edge(hnswNode* f, hnswNode* t, unsigned short dist, unsigned int level):
    cdef hnsw_edge* edge = <hnsw_edge*> malloc(sizeof(hnsw_edge))
    edge.node_id = t.id
    edge.dist = dist
    edge.next = NULL

    cdef hnsw_edge_set* edge_set = f.edges[level]
    if edge_set.head_ptr == NULL:
        edge_set.head_ptr = edge
        edge_set.last_ptr = edge
    else:
        edge_set.last_ptr.next = edge
        edge_set.last_ptr = edge

    edge_set.size += 1


cdef bytes _c2bytes(unsigned char* data):
     cdef unsigned short datalen = len(data)
     cdef bytes retval = PyBytes_FromStringAndSize(NULL, datalen*8)

     cdef char* resbuf = retval # no copy
     cdef unsigned char byte
     cdef unsigned short pos, i
     cdef char* s01 = "01"
     for i in range(datalen):
         byte = data[i]
         for pos in range(8):
             resbuf[8*i + (7-pos)] = s01[(byte >> pos) & 1]
     return retval


cpdef unsigned short hamming_dist(unsigned char *x, unsigned char *y):
     cdef bytes x_bytes = _c2bytes(x)
     cdef bytes y_bytes = _c2bytes(y)

     cdef unsigned short N = len(x_bytes)
     cdef unsigned short i = 0
     cdef unsigned short count = 0

     for i in range(N):
         count += (x_bytes[i] != y_bytes[i])
     return count

cdef class IndexHnsw:
    cdef Pool mem
    cdef Counter cnt
    cdef unsigned short bytes_per_vector
    cdef unsigned int max_level
    cdef prehash_map* nodes_ptr
    cdef hnswNode* entry_ptr

    cpdef void index(self, unsigned int id, unsigned char* vector):
        self._add_node(id, vector)

    cdef hnswNode* _get_node(self, UIDX id):
        return <hnswNode*> prehash_get(self.nodes_ptr, id)

    cdef void _add_node(self, UIDX id, unsigned char*key):
        cdef hnswNode *node
        cdef hnswNode* entry_ptr = self.entry_ptr
        self.cnt.num_total += 1

        if entry_ptr == NULL:
            node = create_node(id, 0, key)
            self.entry_ptr = node
            prehash_insert(self.nodes_ptr, id, node)
            return


        cdef unsigned int level = self._random_level()
        node = create_node(id, level, key)
        prehash_insert(self.nodes_ptr, id, node)

        cdef unsigned short min_dist = hamming_dist(key, self.entry_ptr.key)

        cdef int l = self.max_level

        while l > level:
            entry_id, min_dist = self.greedy_closest_neighbor(key, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1


        l = min(self.max_level, level)

        while l >= 0:
            neighbors = self.search_level(key, entry_ptr, l)

            while neighbors.size > 0:
                  d, item = neighbors.pop()
                  neighbor = self._get_node(item[0])
                  entry_ptr = neighbor

                  _add_edge(node, neighbor, d, l)
                  _add_edge(neighbor, node, d, l)

            l -= 1
        if level > self.max_level:
            self.max_level = level
            self.entry_ptr = node


    cdef search_level(self, unsigned char *query, hnswNode *entry_ptr, unsigned int level):
        cdef unsigned short dist = hamming_dist(query, entry_ptr.key)

        candidate_nodes = PriorityQueue()
        result_nodes = PriorityQueue()
        candidate_nodes.push((entry_ptr.id,), dist) # min priority queue
        result_nodes.push((entry_ptr.id,), 1/(dist+0.1)) # max priority queue

        cdef cpp_set[UIDX] visited_nodes
        cdef hnswNode *candidate
        visited_nodes.insert(entry_ptr.id)
        cdef unsigned short lower_bound = dist

        cdef hnsw_edge* next_edge
        cdef hnsw_edge_set* edge_set

        while not candidate_nodes.empty():
            priority, item = candidate_nodes.pop()
            candidate = self._get_node(item[0])

            if priority > lower_bound:
                break

            edge_set = candidate.edges[level]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                id = next_edge.node_id

                if visited_nodes.find(id) != visited_nodes.end():
                    next_edge = next_edge.next
                    continue

                visited_nodes.insert(id)
                neighbor = self._get_node(id)

                _dist = hamming_dist(query, neighbor.key)
                if _dist < lower_bound or result_nodes.size < 100:
                    candidate_nodes.push((id,), _dist)
                    result_nodes.push((id,), 1/(_dist+0.1))
                    if _dist < lower_bound:
                        lower_bound = _dist

                    if result_nodes.size > 100:
                       result_nodes.pop()

                next_edge = next_edge.next


        return result_nodes

    cdef tuple greedy_closest_neighbor(self, unsigned char *query, hnswNode *entry_ptr, unsigned short min_dist, unsigned int level):
        cdef unsigned short _min_dist = min_dist
        cdef unsigned short dist
        cdef UIDX _entry_id = entry_ptr.id
        cdef hnswNode *node_ptr
        cdef hnswNode *closest_neighbor

        cdef hnsw_edge_set* edge_set
        cdef hnsw_edge* next_edge

        while True:
            closest_neighbor = NULL
            edge_set = entry_ptr.edges[level]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                id = next_edge.node_id

                node_ptr = self._get_node(id)
                dist = hamming_dist(query, node_ptr.key)
                if dist < min_dist:
                    min_dist = dist
                    closest_neighbor = node_ptr

                next_edge = next_edge.next

            if closest_neighbor == NULL:
                break

            entry_ptr = closest_neighbor

        return (entry_ptr.id, _min_dist)


    cdef unsigned int _random_level(self):
        cdef double r = rand() / RAND_MAX
        cdef double f = floor(-log(r) * 6)

        return int(f)

    cpdef query(self, unsigned char *query):
        #cdef array.array final_result = array.array('L')
        #cdef array.array final_idx = array.array('L')

        cdef hnswNode* entry_ptr = self.entry_ptr
        cdef unsigned short min_dist = hamming_dist(query, entry_ptr.key)
        cdef unsigned int l = self.max_level
        cdef UIDX entry_id = entry_ptr.id
        while l > 0:
            entry_id, min_dist = self.greedy_closest_neighbor(query, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1

        neighbors = self.search_level(query, entry_ptr, 0)


        #result_size = 0
        result = []
        while not neighbors.empty():
            dist, item = neighbors.pop()
            node = self._get_node(item[0])
            result.append({
                'id': item[0],
                'vector': node.key,
                'distance': dist
            })

        return result

    cpdef batch_query(self, unsigned char *query, const UIDX num_query):
        cdef UIDX _0
        cdef unsigned short _1
        cdef unsigned char *q_key = <unsigned char*> self.mem.alloc(self.bytes_per_vector, 8)
        result = []
        for _0 in range(num_query):
            for _1 in range(self.bytes_per_vector):
                q_key[_1] = query[_1]
            q_result = self.query(q_key)
            result.append(q_result)

            query += self.bytes_per_vector

        return result

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
        self.entry_ptr = NULL
        self.max_level = 0
        self.nodes_ptr = <prehash_map*> malloc(sizeof(prehash_map))

    def __dealloc__(self):
        # self.free_trie()
        pass

    cpdef destroy(self):
        pass