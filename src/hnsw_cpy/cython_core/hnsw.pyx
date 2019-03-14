# cython: language_level=3, wraparound=False, boundscheck=False

# noinspection PyUnresolvedReferences
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport array

from libc.stdlib cimport rand, RAND_MAX
from libc.string cimport memcpy, strcpy
from libc.math cimport log, floor, abs

from hnsw_cpy.cython_core.heappq cimport heappq, pq_entity, init_heappq, free_heappq, heappq_push, heappq_pop_min, heappq_peak_min, heappq_pop_max, heappq_peak_max
from hnsw_cpy.cython_core.queue cimport queue, init_queue, queue_push_tail, queue_pop_head, queue_free, queue_is_empty


cdef hnswNode* create_node(UIDX id, USHORT level, BVECTOR vector, USHORT bytes_num):
     cdef hnswNode *node = <hnswNode*> PyMem_Malloc(sizeof(hnswNode))
     node.id = id
     node.level = level
     node.next = NULL
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

     # PyMem_Free(vector)

     return node

cdef void _add_edge(hnswNode* f, hnswNode* t, DIST dist, UINT level):
    cdef hnsw_edge* edge = <hnsw_edge*> PyMem_Malloc(sizeof(hnsw_edge))
    edge.node = t
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
        head_edge.node = NULL
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
    cdef USHORT i, dist = 0
    cdef UCHAR byte_x, byte_y, byte_z

    for i in range(datalen):
        byte_x = x[i]
        byte_y = y[i]
        byte_z = byte_x ^ byte_y

        if byte_z > 0:
            dist += 1
    return dist


cdef void _free_node(hnswNode* node):
    cdef hnswNode* next_node = node
    cdef hnswNode* cur_node
    cdef USHORT level
    cdef USHORT l
    while next_node != NULL:
        level = next_node.level
        for l in range(level+1):
            _empty_edge_set(next_node, l)
            PyMem_Free(next_node.edges[l])
            next_node.edges[l] = NULL

        PyMem_Free(next_node.edges)
        next_node.edges = NULL
        PyMem_Free(next_node.vector)
        next_node.vector = NULL
        cur_node = next_node
        next_node = next_node.next
        PyMem_Free(cur_node)

cdef class IndexHnsw:
    cdef hnswConfig* config
    cdef UINT total_size
    cdef USHORT bytes_num
    cdef USHORT max_level
    cdef hnswNode* entry_ptr


    cpdef void index(self, UIDX id, BVECTOR vector):
        self._add_node(id, vector)

    cdef void _add_node(self, UIDX id, BVECTOR vector):
        """Add node to the hnsw graph"""

        cdef hnswNode* new_node
        cdef hnswNode* entry_ptr = self.entry_ptr
        self.total_size += 1

        # create the root node at level 0
        if entry_ptr == NULL:
            new_node = create_node(id, 0, vector, self.bytes_num)
            self.entry_ptr = new_node

            return

        # the HNSW is not empty, we have an entry point

        # level at which the node will be added
        cdef USHORT level = self._random_level()
        new_node = create_node(id, level, vector, self.bytes_num)

        cdef DIST min_dist = hamming_dist(vector, self.entry_ptr.vector, self.bytes_num)

        cdef int l = self.max_level

        cdef hnsw_edge* result_item
        while l > level:
            # search for the closest neighbor
            result_item = self.greedy_closest_neighbor(vector, entry_ptr, min_dist, l)
            entry_ptr = result_item.node
            min_dist = result_item.dist

            PyMem_Free(result_item)

            l -= 1


        l = min(self.max_level, level)
        cdef hnswNode* neighbor
        cdef UINT m_max
        cdef DIST dist
        cdef heappq* neighbors_pq
        cdef heappq* selected_pq
        cdef pq_entity* pq_e

        while l >= 0:
            # navigate the graph and create edges with the closest nodes we find
            neighbors_pq = self.search_level(vector, entry_ptr, self.config.ef_construction, l)
            selected_pq = self._select_neighbors(vector, neighbors_pq, self.config.m, l, True)
            free_heappq(neighbors_pq)

            pq_e = heappq_peak_min(selected_pq)
            if pq_e.priority == 0:
                neighbor = <hnswNode*> pq_e.value
                new_node.next = neighbor.next
                neighbor.next = new_node
                level = neighbor.level
                free_heappq(selected_pq)
                break

            while selected_pq.size > 0:
                pq_e = heappq_pop_max(selected_pq)
                dist = pq_e.priority
                neighbor = <hnswNode*> pq_e.value
                pq_e.value = NULL
                PyMem_Free(pq_e)

                entry_ptr = neighbor

                _add_edge(new_node, neighbor, dist, l)
                _add_edge(neighbor, new_node, dist, l)

                m_max = self.config.m_max
                if l == 0:
                    m_max = self.config.m_max_0
                if neighbor.edges[l].size > m_max:
                    self._prune_neighbors(neighbor, m_max, l)



            l -= 1
            free_heappq(selected_pq)

        if level > self.max_level:
            self.max_level = level
            self.entry_ptr = new_node


    cdef heappq* search_level(self, BVECTOR query, hnswNode *entry_ptr, UINT ef, USHORT level):
        cdef DIST dist = hamming_dist(query, entry_ptr.vector, self.bytes_num)

        cdef heappq* candidates_pq = init_heappq()
        cdef heappq* result_pq = init_heappq()
        heappq_push(candidates_pq, dist, entry_ptr)
        heappq_push(result_pq, dist, entry_ptr)

        cdef set visited_nodes = set()
        visited_nodes.add(entry_ptr.id)

        cdef hnswNode* candidate
        cdef hnswNode* neighbor

        cdef DIST lower_bound

        cdef hnsw_edge* next_edge
        cdef hnsw_edge_set* edge_set
        cdef UIDX id

        cdef pq_entity* pq_e
        cdef pq_entity* _e

        while candidates_pq.size > 0:
            pq_e = heappq_pop_min(candidates_pq)
            priority = pq_e.priority
            candidate = <hnswNode*> pq_e.value
            pq_e.value = NULL
            PyMem_Free(pq_e)

            lower_bound = heappq_peak_max(result_pq).priority

            if priority > lower_bound:
                break

            edge_set = candidate.edges[level]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                neighbor = next_edge.node

                if neighbor.id in visited_nodes:
                    next_edge = next_edge.next
                    continue

                visited_nodes.add(neighbor.id)

                dist = hamming_dist(query, neighbor.vector, self.bytes_num)

                if dist < lower_bound or result_pq.size < ef:
                    heappq_push(candidates_pq, dist, neighbor)
                    heappq_push(result_pq, dist, neighbor)

                    if result_pq.size > ef:
                        _e = heappq_pop_max(result_pq)
                        _e.value = NULL
                        PyMem_Free(_e)

                # elif dist == lower_bound:
                #    heappq_push(candidates_pq, dist, neighbor)

                next_edge = next_edge.next

        visited_nodes.clear()

        free_heappq(candidates_pq)
        candidates_pq = NULL

        return result_pq

    cdef hnsw_edge* greedy_closest_neighbor(self, BVECTOR query, hnswNode *entry_ptr, DIST min_dist, USHORT level):
        cdef DIST _min_dist = min_dist
        cdef DIST dist

        cdef hnswNode *node_ptr

        cdef hnswNode *closest_neighbor = entry_ptr

        cdef hnsw_edge_set* edge_set
        cdef hnsw_edge* next_edge
        cdef set visited_nodes = set()

        cdef queue* candidates = init_queue()
        queue_push_tail(candidates, entry_ptr)
        visited_nodes.add(entry_ptr.id)

        while not queue_is_empty(candidates):
            node_ptr = <hnswNode*> queue_pop_head(candidates)

            edge_set = node_ptr.edges[level]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                node_ptr = next_edge.node

                if node_ptr.id in visited_nodes:
                    next_edge = next_edge.next
                    continue
                visited_nodes.add(node_ptr.id)

                dist = hamming_dist(query, node_ptr.vector, self.bytes_num)
                if dist < _min_dist:
                    _min_dist = dist
                    closest_neighbor = node_ptr
                    while not queue_is_empty(candidates):
                        queue_pop_head(candidates)
                    queue_push_tail(candidates, node_ptr)

                elif dist == _min_dist:
                    queue_push_tail(candidates, node_ptr)


                next_edge = next_edge.next

        visited_nodes.clear()
        queue_free(candidates)

        cdef hnsw_edge* edge = <hnsw_edge*> PyMem_Malloc(sizeof(hnsw_edge))
        edge.node = closest_neighbor
        edge.dist = _min_dist
        return edge


    cdef heappq* _select_neighbors(self, BVECTOR query, heappq* neighbors_pq, USHORT ensure_k, USHORT level, bint extend_candidates):
        cdef heappq* result_pq = init_heappq()
        cdef set visited_nodes = set()

        cdef hnsw_edge* next_edeg
        cdef hnsw_edge_set* edge_set
        cdef DIST dist
        cdef pq_entity* pq_e

        if extend_candidates or neighbors_pq.size < ensure_k:
            while neighbors_pq.size > 0:
                pq_e = heappq_pop_min(neighbors_pq)
                priority = pq_e.priority
                candidate = <hnswNode*> pq_e.value
                pq_e.value = NULL
                PyMem_Free(pq_e)

                if candidate.id in visited_nodes:
                    continue

                heappq_push(result_pq, priority, candidate)
                visited_nodes.add(candidate.id)

                edge_set = candidate.edges[level]
                next_edge = edge_set.head_ptr
                while next_edge != NULL:
                    candidate = next_edge.node

                    if candidate.id in visited_nodes:
                        next_edge = next_edge.next
                        continue

                    visited_nodes.add(candidate.id)

                    dist = hamming_dist(query, candidate.vector, self.bytes_num)

                    heappq_push(result_pq, dist, candidate)
                    next_edge = next_edge.next

        else:
            while neighbors_pq.size > 0 and result_pq.size < ensure_k:
                pq_e = heappq_pop_min(neighbors_pq)
                candidate = <hnswNode*> pq_e.value
                priority = pq_e.priority
                heappq_push(result_pq, priority, candidate)
                pq_e.value = NULL
                PyMem_Free(pq_e)


        while result_pq.size > ensure_k:
            pq_e = heappq_pop_max(result_pq)
            pq_e.value = NULL
            PyMem_Free(pq_e)

        visited_nodes.clear()

        return result_pq

    cdef void _prune_neighbors(self, hnswNode* node, UINT k, USHORT level):
        cdef heappq* neighbors_pq = init_heappq()
        cdef heappq* selected_pq

        cdef hnsw_edge_set* edge_set = node.edges[level]
        cdef hnsw_edge* next_edge = edge_set.head_ptr

        cdef DIST dist
        cdef hnswNode* neighbor
        while next_edge != NULL:
            neighbor = next_edge.node
            dist = next_edge.dist

            heappq_push(neighbors_pq, dist, neighbor)

            next_edge = next_edge.next

        selected_pq = self._select_neighbors(node.vector, neighbors_pq, k, level, True)
        free_heappq(neighbors_pq)

        _empty_edge_set(node, level)
        cdef pq_entity* pq_e
        while selected_pq.size > 0:
            pq_e = heappq_pop_min(selected_pq)
            dist = pq_e.priority
            neighbor = <hnswNode*> pq_e.value
            pq_e.value = NULL
            PyMem_Free(pq_e)
            _add_edge(node, neighbor, dist, level)

        free_heappq(selected_pq)

    cdef USHORT _random_level(self):
        cdef double r = rand() / RAND_MAX
        cdef double f = floor(-log(r) * self.config.level_multiplier)

        return int(f)

    cpdef list query(self, BVECTOR query, USHORT top_k):
        cdef hnswNode* entry_ptr = self.entry_ptr

        cdef DIST min_dist = hamming_dist(query, entry_ptr.vector, self.bytes_num)
        cdef int l = self.max_level
        cdef hnsw_edge* result_item
        while l > 0:
            result_item = self.greedy_closest_neighbor(query, entry_ptr, min_dist, l)
            entry_ptr = result_item.node
            min_dist = result_item.dist
            PyMem_Free(result_item)

            l -= 1


        cdef UINT ef = max(self.config.ef, top_k)
        cdef heappq* neighbors_pq = self.search_level(query, entry_ptr, ef, 0)
        cdef USHORT count = 0

        cdef list result = []

        cdef pq_entity* pq_e
        cdef hnswNode* next_node
        cdef DIST dist
        while neighbors_pq.size > 0:
            pq_e = heappq_pop_min(neighbors_pq)
            dist = pq_e.priority
            next_node = <hnswNode*> pq_e.value
            pq_e.value = NULL
            PyMem_Free(pq_e)
            while next_node != NULL:
                if count >= top_k:
                    break
                result.append({
                    'id': next_node.id,
                    'distance': dist
                })
                count += 1
                next_node = next_node.next

            if count >= top_k:
                break

        free_heappq(neighbors_pq)
        neighbors_pq = NULL
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


    cdef queue* _get_nodes(self):
        cdef set visited_nodes = set()
        cdef queue* nodes_queue = init_queue()
        cdef queue* candidates_queue = init_queue()

        cdef hnswNode* node_ptr = self.entry_ptr
        cdef hnswNode* neighbor
        cdef hnsw_edge_set* edge_set
        cdef hnsw_edge* next_edge

        queue_push_tail(candidates_queue, node_ptr)
        visited_nodes.add(node_ptr.id)

        while not queue_is_empty(candidates_queue):
            node_ptr = <hnswNode*> queue_pop_head(candidates_queue)

            edge_set = node_ptr.edges[0]
            next_edge = edge_set.head_ptr

            while next_edge != NULL:
                neighbor = next_edge.node
                if neighbor == NULL:
                    next_edge = next_edge.next
                    continue

                if neighbor.id in visited_nodes:
                    next_edge = next_edge.next
                    continue

                queue_push_tail(candidates_queue, neighbor)
                visited_nodes.add(neighbor.id)

                next_edge = next_edge.next

            queue_push_tail(nodes_queue, node_ptr)

        queue_free(candidates_queue)
        visited_nodes.clear()

        return nodes_queue

    cpdef void save_model(self, model_path):
        pass


    cpdef void load_model(self, model_path):
        pass

    cdef void free_hnsw(self):
        cdef queue* nodes_queue = self._get_nodes()
        while not queue_is_empty(nodes_queue):
            node_ptr = <hnswNode*> queue_pop_head(nodes_queue)
            _free_node(node_ptr)
        queue_free(nodes_queue)

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
        self.free_hnsw()

    cpdef destroy(self):
        self.free_hnsw()
