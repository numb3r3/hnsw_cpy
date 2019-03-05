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


from hnsw_cpy.cython_core.utils import PriorityQueue

cdef hnswNode* create_node(UIDX id, USHORT level, BVECTOR vector, USHORT bytes_per_vector):
     cdef hnswNode *node = <hnswNode*> PyMem_Malloc(sizeof(hnswNode))
     node.id = id
     node.level = level
     cdef USHORT N = bytes_per_vector * sizeof(UCHAR) + 1 # +1 for the null-terminator

     node.vector = <BVECTOR> PyMem_Malloc(N)
     memcpy(node.vector, vector, N)

     node.edges = <hnsw_edge_set**> PyMem_Malloc((level+1) * sizeof(hnsw_edge_set*))

     cdef hnsw_edge_set* edge_set
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


cdef void _empty_edge_set(hnswNode* node, UINT level):
    cdef hnsw_edge_set* edge_set = node.edges[level]
    cdef hnsw_edge* head_edge = edge_set.head_ptr
    while head_edge != NULL:
        edge_set.head_ptr = head_edge.next
        PyMem_Free(head_edge)
        head_edge = edge_set.head_ptr
    edge_set.head_ptr = NULL
    edge_set.last_ptr = NULL

    # PyMem_Free(edge_set)


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


cpdef USHORT hamming_dist(BVECTOR x, BVECTOR y, USHORT bytes_per_vector):
     cdef bytes x_bytes = _c2bytes(x, bytes_per_vector)
     cdef bytes y_bytes = _c2bytes(y, bytes_per_vector)

     cdef USHORT N = bytes_per_vector * 8
     cdef USHORT i = 0
     cdef USHORT count = 0

     for i in range(N):
         count += (x_bytes[i] != y_bytes[i])
     return count

cdef void _free_node(hnswNode* node):
    cdef USHORT level = node.level
    cdef USHORT _0
    for _0 in range(level):
        _empty_edge_set(node, _0)
        PyMem_Free(node.edges[_0])
        node.edges[_0] = NULL

    PyMem_Free(node.edges)
    node.edges = NULL
    PyMem_Free(node.vector)
    node.vector = NULL
    PyMem_Free(node)

cdef class IndexHnsw:
    cdef hnswConfig* config
    cdef UINT total_size
    cdef USHORT bytes_per_vector
    cdef USHORT max_level
    cdef nodes_map nodes
    cdef hnswNode* entry_ptr

    cpdef void index(self, UIDX id, BVECTOR vector):
        self._add_node(id, vector)

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

    cdef void _add_node(self, UIDX id, BVECTOR vector):
        cdef hnswNode *new_node
        cdef hnswNode* entry_ptr = self.entry_ptr
        self.total_size += 1

        if entry_ptr == NULL:
            new_node = create_node(id, 0, vector, self.bytes_per_vector)
            self.entry_ptr = new_node
            self._insert_node(new_node)

            return

        cdef USHORT level = self._random_level()
        new_node = create_node(id, level, vector, self.bytes_per_vector)
        self._insert_node(new_node)

        cdef DIST min_dist = hamming_dist(vector, self.entry_ptr.vector, self.bytes_per_vector)

        cdef int l = self.max_level

        while l > level:
            entry_id, min_dist = self.greedy_closest_neighbor(vector, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1


        l = min(self.max_level, level)
        cdef hnswNode* neighbor
        cdef UINT m_max

        while l >= 0:
            neighbors = self.search_level(vector, entry_ptr, self.config.ef_construction, l)

            neighbors = self._select_neighbors(vector, neighbors, self.config.m, l, True)

            while neighbors.size > 0:
                dist, item = neighbors.pop()
                neighbor = self._get_node(item[0])
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


    cdef search_level(self, BVECTOR query, hnswNode *entry_ptr, UINT ef, USHORT level):

        cdef DIST dist = hamming_dist(query, entry_ptr.vector, self.bytes_per_vector)

        candidate_nodes = PriorityQueue()
        result_nodes = PriorityQueue()
        candidate_nodes.push((entry_ptr.id,), dist) # min priority queue
        result_nodes.push((entry_ptr.id,), 1/(dist+1)) # max priority queue

        cdef cpp_set[UIDX] visited_nodes

        cdef hnswNode* candidate
        visited_nodes.insert(entry_ptr.id)

        cdef DIST lower_bound = dist

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

                lb = visited_nodes.lower_bound(id)
                if lb != visited_nodes.end() and id == deref(lb):
                    next_edge = next_edge.next
                    continue

                #if visited_nodes.find(id) != visited_nodes.end():
                #    next_edge = next_edge.next
                #    continue

                visited_nodes.insert(id)
                neighbor = self._get_node(id)

                dist = hamming_dist(query, neighbor.vector, self.bytes_per_vector)

                # TODO: add hnswConfig
                if dist < lower_bound or result_nodes.size < ef:
                    candidate_nodes.push((id,), dist)
                    result_nodes.push((id,), 1/(dist+1.0))

                    if dist < lower_bound:
                        lower_bound = dist

                    if result_nodes.size > ef:
                       result_nodes.pop()

                next_edge = next_edge.next

        visited_nodes.clear()
        return result_nodes

    cdef tuple greedy_closest_neighbor(self, BVECTOR query, hnswNode *entry_ptr, DIST  min_dist, USHORT level):
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

                dist = hamming_dist(query, node_ptr.vector, self.bytes_per_vector)
                if dist < min_dist:
                    min_dist = dist
                    closest_neighbor = node_ptr

                next_edge = next_edge.next

            if closest_neighbor == NULL:
                break

            entry_ptr = closest_neighbor

        return (entry_ptr.id, _min_dist)


    cdef _select_neighbors(self, BVECTOR query, neighbors, USHORT ensure_k, USHORT level, bint extend_candidates):
        # reverse neighors
        candidates = PriorityQueue()
        neighbors_clone = PriorityQueue()

        cdef cpp_set[UIDX] existing_candidates
        while not neighbors.empty():
            priority, item = neighbors.pop()
            candidates.push((item[0],), 1 / priority - 1)
            neighbors_clone.push((item[0],), priority)
            existing_candidates.insert(item[0])

        cdef USHORT candidates_size = neighbors.size

        cdef hnsw_edge* next_edeg
        cdef hnsw_edge_set* edge_set
        cdef DIST dist

        if extend_candidates:
            while not neighbors_clone.empty():
                priority, item = neighbors_clone.pop()
                candidate = self._get_node(item[0])

                edge_set = candidate.edges[level]
                next_edge = edge_set.head_ptr
                while next_edge != NULL:
                    id = next_edge.node_id
                    lb = existing_candidates.lower_bound(id)
                    if lb != existing_candidates.end() and id == deref(lb):
                        next_edge = next_edge.next
                        continue
                    #if existing_candidates.find(id) != existing_candidates.end():
                    #    next_edge = next_edge.next
                    #    continue
                    existing_candidates.insert(id)
                    candidate = self._get_node(id)

                    dist = hamming_dist(query, candidate.vector, self.bytes_per_vector)
                    candidates.push((id,), dist)
        result = PriorityQueue()
        while (not candidates.empty()) and result.size < ensure_k:
            p, t = candidates.pop()
            result.push((t[0],), p)

        existing_candidates.clear()

        return result

    cdef void _prune_neighbors(self, hnswNode* node, UINT k, USHORT level):
        neighbors = PriorityQueue()
        cdef hnsw_edge_set* edge_set = node.edges[level]
        cdef hnsw_edge* next_edge = edge_set.head_ptr
        cdef UIDX node_id
        cdef DIST dist
        cdef hnswNode* neighbor
        while next_edge != NULL:
            node_id = next_edge.node_id
            dist = next_edge.dist

            neighbors.push((node_id,), 1 / (dist+1.0))
            next_edge = next_edge.next

        neighbors = self._select_neighbors(node.vector, neighbors, self.config.m, level, True)

        _empty_edge_set(node, level)
        while not neighbors.empty():
            dist, item = neighbors.pop()
            neighbor = self._get_node(item[0])
            _add_edge(node, neighbor, dist, level)

    cdef USHORT _random_level(self):
        cdef double r = rand() / RAND_MAX
        cdef double f = floor(-log(r) * self.config.level_multiplier)

        return int(f)

    cpdef list query(self, BVECTOR query, USHORT top_k):
        cdef hnswNode* entry_ptr = self.entry_ptr

        cdef DIST min_dist = hamming_dist(query, entry_ptr.vector, self.bytes_per_vector)
        cdef USHORT l = self.max_level
        cdef UIDX entry_id = entry_ptr.id
        while l > 0:
            entry_id, min_dist = self.greedy_closest_neighbor(query, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1

        cdef UINT ef = max(self.config.ef, top_k)
        neighbors = self.search_level(query, entry_ptr, ef, 0)

        cdef UINT cutoff = 0
        if neighbors.size > top_k:
            cutoff = neighbors.size - top_k

        cdef list result = []
        while not neighbors.empty():
            dist, item = neighbors.pop()
            if cutoff > 0:
                cutoff -= 1
                continue
            node = self._get_node(item[0])
            result.append({
                'id': item[0],
                # 'vector': <bytes> node.vector,
                'distance': 1.0/dist-1
            })

        return result[::-1]

    cpdef batch_query(self, BVECTOR query, const USHORT num_query, const USHORT k):
        cdef UIDX _0
        cdef USHORT _1
        cdef BVECTOR q_key = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * self.bytes_per_vector)
        result = []
        for _0 in range(num_query):
            for _1 in range(self.bytes_per_vector):
                q_key[_1] = query[_1]
            q_result = self.query(q_key, k)
            result.append(q_result)

            query += self.bytes_per_vector

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


    def __cinit__(self, bytes_per_vector, **kwargs):
        self.bytes_per_vector = bytes_per_vector

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
