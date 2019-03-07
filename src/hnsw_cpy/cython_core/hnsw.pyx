# cython: language_level=3, wraparound=False, boundscheck=False

# noinspection PyUnresolvedReferences
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport array
# from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.string cimport memcpy, strcpy
from libc.math cimport log, floor

from libcpp.set cimport set as cpp_set

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from cython.operator cimport postincrement


# from hnsw_cpy.cython_core.utils import PriorityQueue
from hnsw_cpy.cython_core.heappq cimport heappq, pq_entity, init_heappq, free_heappq, heappq_push, heappq_pop_min, heappq_peak_min, heappq_pop_max, heappq_peak_max


cdef hnswNode* create_node(UIDX id, USHORT level, BVECTOR vector, USHORT bytes_num):
     cdef hnswNode *node = <hnswNode*> PyMem_Malloc(sizeof(hnswNode))
     node.id = id
     node.level = level
     cdef USHORT N = bytes_num * sizeof(UCHAR) + 1 # +1 for the null-terminator

     node.vector = <BVECTOR> PyMem_Malloc(N)
     memcpy(node.vector, vector, N)

     node.edges = <hnsw_edge_set**> PyMem_Malloc((level+1) * sizeof(hnsw_edge_set*))

     cdef hnsw_edge_set* edge_set
     cdef USHORT l
     for l in range(level+1):
         edge_set = <hnsw_edge_set*> PyMem_Malloc(sizeof(hnsw_edge_set))
         edge_set.size = 0
         edge_set.head_ptr = NULL
         edge_set.last_ptr = NULL

         node.edges[l] = edge_set

     return node

cdef void _add_edge(hnswNode* f, hnswNode* t, DIST dist, UINT level):
    cdef hnsw_edge* edge = <hnsw_edge*> PyMem_Malloc(sizeof(hnsw_edge))
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


cdef void _empty_edge_set(hnswNode* node, USHORT level):
    cdef hnsw_edge_set* edge_set = node.edges[level]
    cdef hnsw_edge* head_edge = edge_set.head_ptr
    while head_edge != NULL:
        edge_set.head_ptr = head_edge.next
        PyMem_Free(head_edge)
        head_edge = edge_set.head_ptr
    edge_set.size = 0
    edge_set.head_ptr = NULL
    edge_set.last_ptr = NULL

cdef bytes _c2bytes(BVECTOR data, USHORT datalen):
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

cpdef USHORT hamming_dist(BVECTOR x, BVECTOR y, USHORT datalen):
    cdef USHORT i, dist
    cdef UCHAR byte_x, byte_y, byte_z

    for i in range(datalen):
        byte_x = x[i]
        byte_y = y[i]
        byte_z = byte_x ^ byte_y
        while byte_z > 0:
            dist += 1
            byte_z &= byte_z - 1
    return dist


cdef void _free_node(hnswNode* node):
    cdef USHORT level = node.level
    cdef USHORT l
    for l in range(level+1):
        _empty_edge_set(node, l)
        PyMem_Free(node.edges[l])
        node.edges[l] = NULL

    PyMem_Free(node.edges)
    node.edges = NULL
    PyMem_Free(node.vector)
    node.vector = NULL
    PyMem_Free(node)

cdef class IndexHnsw:
    cdef hnswConfig* config
    cdef UINT total_size
    cdef USHORT bytes_num
    cdef USHORT max_level
    cdef nodes_map nodes
    cdef hnswNode* entry_ptr

    cdef hnswNode* _get_node(self, UIDX id):
        lb = self.nodes.lower_bound(id)
        # Check if the key already exists
        if lb != self.nodes.end() and id == deref(lb).first:
            return deref(lb).second

    cdef void _insert_node(self, hnswNode* node):
        cdef UIDX key = node.id
        lb = self.nodes.lower_bound(key)
        cdef node_item item = node_item(key, node)
        self.nodes.insert(lb, item)


    cpdef void index(self, UIDX id, BVECTOR vector):
        self._add_node(id, vector)

    cdef void _add_node(self, UIDX id, BVECTOR vector):
        cdef hnswNode* new_node
        cdef hnswNode* entry_ptr = self.entry_ptr
        self.total_size += 1

        if entry_ptr == NULL:
            # create the root node at level 0
            new_node = create_node(id, 0, vector, self.bytes_num)
            self.entry_ptr = new_node
            self._insert_node(new_node)

            return

        cdef USHORT level = self._random_level()
        new_node = create_node(id, level, vector, self.bytes_num)
        self._insert_node(new_node)

        cdef DIST min_dist = hamming_dist(vector, self.entry_ptr.vector, self.bytes_num)

        cdef int l = self.max_level

        cdef UIDX entry_id
        while l > level:
            entry_id, min_dist = self.greedy_closest_neighbor(vector, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1

        l = min(self.max_level, level)
        cdef hnswNode* neighbor
        cdef UINT m_max
        cdef DIST dist
        cdef heappq* neighbors_pq
        cdef pq_entity* pq_e

        while l >= 0:
            neighbors_pq = self.search_level(vector, entry_ptr, self.config.ef_construction, l)

            neighbors_pq = self._select_neighbors(vector, neighbors_pq, self.config.m, l, True)

            neighbor = NULL
            while neighbors_pq.size > 0:
                # dist, item = neighbors.pop()
                pq_e = heappq_pop_min(neighbors_pq)
                dist = pq_e.priority
                neighbor = <hnswNode*> pq_e.value

                # neighbor = self._get_node(item[0])

                # the first pop is nearest neighbor
                if neighbor == NULL:
                    entry_ptr = neighbor

                _add_edge(new_node, neighbor, dist, l)
                _add_edge(neighbor, new_node, dist, l)

                m_max = self.config.m_max
                if l == 0:
                    m_max = self.config.m_max_0
                if neighbor.edges[l].size > m_max:
                    self._prune_neighbors(neighbor, m_max, l)

            l -= 1

        if level > self.max_level:
            self.max_level = level
            self.entry_ptr = new_node


    cdef heappq* search_level(self, BVECTOR query, hnswNode *entry_ptr, UINT ef, USHORT level):
        cdef DIST dist = hamming_dist(query, entry_ptr.vector, self.bytes_num)

        cdef heappq* candidates_pq = init_heappq()
        cdef heappq* result_pq = init_heappq()
        heappq_push(candidates_pq, dist, entry_ptr)
        heappq_push(result_pq, dist, entry_ptr)

        cdef cpp_set[UIDX] visited_nodes

        cdef hnswNode* candidate
        visited_nodes.insert(entry_ptr.id)

        cdef DIST lower_bound

        cdef hnsw_edge* next_edge
        cdef hnsw_edge_set* edge_set
        cdef UIDX id

        cdef pq_entity* pq_e
        cdef DIST d
        while candidates_pq.size > 0:
            pq_e = heappq_pop_min(candidates_pq)
            priority = pq_e.priority
            candidate = <hnswNode*> pq_e.value

            lower_bound = heappq_peak_max(result_pq).priority

            if priority > lower_bound:
                break

            edge_set = candidate.edges[level]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                id = next_edge.node_id
                neighbor = self._get_node(id)

                lb = visited_nodes.lower_bound(id)
                if lb != visited_nodes.end() and id == deref(lb):
                    next_edge = next_edge.next
                    continue

                visited_nodes.insert(id)

                dist = hamming_dist(query, neighbor.vector, self.bytes_num)

                if dist < lower_bound or result_pq.size < ef:
                    heappq_push(candidates_pq, dist, neighbor)
                    heappq_push(result_pq, dist, neighbor)

                    if result_pq.size > ef:
                        heappq_pop_max(result_pq)

                next_edge = next_edge.next

        visited_nodes.clear()

        free_heappq(candidates_pq)
        candidates_pq = NULL

        return result_pq

    cdef tuple greedy_closest_neighbor(self, BVECTOR query, hnswNode *entry_ptr, DIST min_dist, USHORT level):
        cdef DIST _min_dist = min_dist
        cdef DIST dist
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

                dist = hamming_dist(query, node_ptr.vector, self.bytes_num)
                if dist < _min_dist:
                    _min_dist = dist
                    closest_neighbor = node_ptr

                next_edge = next_edge.next

            if closest_neighbor == NULL:
                break

            entry_ptr = closest_neighbor

        return (entry_ptr.id, _min_dist)


    cdef heappq* _select_neighbors(self, BVECTOR query, heappq* neighbors_pq, USHORT ensure_k, USHORT level, bint extend_candidates):
        cdef heappq* candidates_pq = init_heappq()
        cdef cpp_set[UIDX] existing_candidates

        # cdef USHORT candidates_size = neighbors.size

        cdef hnsw_edge* next_edeg
        cdef hnsw_edge_set* edge_set
        cdef DIST dist
        cdef pq_entity* pq_e

        if extend_candidates:
            while neighbors_pq.size > 0:
                pq_e = heappq_pop_min(neighbors_pq)
                priority = pq_e.priority
                candidate = <hnswNode*> pq_e.value
                heappq_push(candidates_pq, priority, candidate)
                existing_candidates.insert(candidate.id)

                edge_set = candidate.edges[level]
                next_edge = edge_set.head_ptr
                while next_edge != NULL:
                    id = next_edge.node_id
                    lb = existing_candidates.lower_bound(id)
                    if lb != existing_candidates.end() and id == deref(lb):
                        next_edge = next_edge.next
                        continue

                    existing_candidates.insert(id)
                    candidate = self._get_node(id)

                    dist = hamming_dist(query, candidate.vector, self.bytes_num)
                    heappq_push(candidates_pq, dist, candidate)
        else:
            candidates_pq = neighbors_pq


        cdef heappq* result_pq = init_heappq()
        while candidates_pq.size > 0 and result_pq.size < ensure_k:
            pq_e = heappq_pop_min(candidates_pq)
            heappq_push(result_pq, pq_e.priority, pq_e.value)


        existing_candidates.clear()
        free_heappq(candidates_pq)

        return result_pq

    cdef void _prune_neighbors(self, hnswNode* node, UINT k, USHORT level):
        cdef heappq* neighbors_pq = init_heappq()

        cdef hnsw_edge_set* edge_set = node.edges[level]
        cdef hnsw_edge* next_edge = edge_set.head_ptr
        cdef UIDX node_id
        cdef DIST dist
        cdef hnswNode* neighbor
        while next_edge != NULL:
            node_id = next_edge.node_id
            dist = next_edge.dist
            neighbor = self._get_node(node_id)

            heappq_push(neighbors_pq, dist, neighbor)

            next_edge = next_edge.next

        neighbors_pq = self._select_neighbors(node.vector, neighbors_pq, self.config.m, level, True)

        _empty_edge_set(node, level)
        cdef pq_entity* pq_e
        while neighbors_pq.size > 0:
            pq_e = heappq_pop_min(neighbors_pq)
            dist = pq_e.priority
            neighbor = <hnswNode*> pq_e.value
            _add_edge(node, neighbor, dist, level)

        free_heappq(neighbors_pq)

    cdef USHORT _random_level(self):
        cdef double r = rand() / RAND_MAX
        cdef double f = floor(-log(r) * self.config.level_multiplier)

        return int(f)

    cpdef list query(self, BVECTOR query, USHORT top_k):
        cdef hnswNode* entry_ptr = self.entry_ptr

        cdef DIST min_dist = hamming_dist(query, entry_ptr.vector, self.bytes_num)
        cdef int l = self.max_level
        cdef UIDX entry_id = entry_ptr.id
        while l > 0:
            entry_id, min_dist = self.greedy_closest_neighbor(query, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1

        cdef UINT ef = max(self.config.ef, top_k)
        cdef heappq* neighbors_pq = self.search_level(query, entry_ptr, ef, 0)
        cdef USHORT count = 0

        cdef list result = []

        cdef pq_entity* pq_e
        cdef hnswNode* node
        cdef DIST dist
        while neighbors_pq.size > 0:
            pq_e = heappq_pop_min(neighbors_pq)
            dist = pq_e.priority
            node = <hnswNode*> pq_e.value
            result.append({
                'id': node.id,
                'distance': dist
            })
            count += 1
            if count >= top_k:
                break

        return result

    cpdef batch_query(self, BVECTOR query, const USHORT num_query, const USHORT k):
        cdef UIDX _0
        cdef USHORT _1
        cdef BVECTOR q_key = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * self.bytes_num)
        result = []
        for _0 in range(num_query):
            for _1 in range(self.bytes_num):
                q_key[_1] = query[_1]
            q_result = self.query(q_key, k)
            result.append(q_result)

            query += self.bytes_num

        PyMem_Free(q_key)

        return result

    cpdef void save_model(self, model_path):
        pass

    cpdef void load_model(self, model_path):
        pass

    cdef void free_hnsw(self):
        it = self.nodes.begin()
        cdef hnswNode* node
        while (it != self.nodes.end()):
            node = deref(it).second
            _free_node(node)
            node = NULL
            postincrement(it)
        self.nodes.clear()
        PyMem_Free(self.config)
        self.config = NULL

        self.entry_ptr = NULL
        self.total_size = 0
        self.max_level = 0


    @property
    def size(self):
        return self.total_size

    @property
    def memory_size(self):
        raise NotImplemented
        #return get_memory_size(self.root_node)


    def __cinit__(self, bytes_num, **kwargs):
        self.bytes_num = bytes_num

        self.config = <hnswConfig*> PyMem_Malloc(sizeof(hnswConfig))
        self.config.level_multiplier = kwargs.get('level_multiplier', -1)
        self.config.ef = kwargs.get('ef', 20)
        self.config.ef_construction = kwargs.get('ef_construction', 150)
        self.config.m = kwargs.get('m', 12)
        self.config.m_max = kwargs.get('m_max', -1)
        self.config.m_max_0 = kwargs.get('m_max_0', -1)

        if self.config.level_multiplier == -1:
            self.config.level_multiplier = 1.0 / log(1.0*self.config.m)

        if self.config.m_max == -1:
            self.config.m_max = self.config.m

        if self.config.m_max_0 == -1:
            self.config.m_max_0 = 2 * self.config.m

        self.entry_ptr = NULL
        self.max_level = 0

    def __dealloc__(self):
        # self.free_trie()
        self.free_hnsw()

    cpdef destroy(self):
        self.free_hnsw()
