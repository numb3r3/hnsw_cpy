# cython: language_level=3, wraparound=False, boundscheck=False

# noinspection PyUnresolvedReferences
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport array
#from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, floor

from libcpp.set cimport set as cpp_set

from hnsw_cpy.cython_hnsw.utils import PriorityQueue
from hnsw_cpy.cython_lib.prehash cimport prehash_map, prehash_insert, prehash_get


cdef hnswNode* create_node(UIDX id, UINT level, BVECTOR vector):
     cdef hnswNode *node = <hnswNode*> PyMem_Malloc(sizeof(hnswNode))
     node.id = id
     node.level = level
     node.vector = vector

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


cdef bytes _c2bytes(BVECTOR data):
     cdef USHORT datalen = len(data)
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


cpdef unsigned short hamming_dist(BVECTOR x, BVECTOR y):
     cdef bytes x_bytes = _c2bytes(x)
     cdef bytes y_bytes = _c2bytes(y)

     cdef USHORT N = len(x_bytes)
     cdef USHORT i = 0
     cdef USHORT count = 0

     for i in range(N):
         count += (x_bytes[i] != y_bytes[i])
     return count

cdef class IndexHnsw:
    cdef UINT total_size
    cdef USHORT bytes_per_vector
    cdef USHORT max_level
    cdef prehash_map* nodes_ptr
    cdef hnswNode* entry_ptr

    cpdef void index(self, UIDX id, BVECTOR vector):
        self._add_node(id, vector)

    cdef hnswNode* _get_node(self, UIDX id):
        return <hnswNode*> prehash_get(self.nodes_ptr, id)

    cdef void _add_node(self, UIDX id, BVECTOR vector):
        cdef hnswNode *new_node
        cdef hnswNode* entry_ptr = self.entry_ptr
        self.total_size += 1

        if entry_ptr == NULL:
            new_node = create_node(id, 0, vector)
            self.entry_ptr = new_node
            prehash_insert(self.nodes_ptr, id, new_node)
            return


        cdef USHORT level = self._random_level()
        new_node = create_node(id, level, vector)
        prehash_insert(self.nodes_ptr, id, new_node)

        cdef DIST dist = hamming_dist(vector, self.entry_ptr.vector)

        cdef short l = self.max_level

        while l > level:
            entry_id, min_dist = self.greedy_closest_neighbor(vector, entry_ptr, min_dist, l)
            entry_ptr = self._get_node(entry_id)
            l -= 1


        l = min(self.max_level, level)
        # cdef DIST _dist
        cdef hnswNode* neighbor

        while l >= 0:
            neighbors = self.search_level(vector, entry_ptr, l)

            neighbors = self._select_neighbors(vector, neighbors, 60, l, True)

            while neighbors.size > 0:
                  dist, item = neighbors.pop()
                  neighbor = self._get_node(item[0])
                  entry_ptr = neighbor

                  _add_edge(new_node, neighbor, dist, l)
                  _add_edge(neighbor, new_node, dist, l)

            l -= 1

        if level > self.max_level:
            self.max_level = level
            self.entry_ptr = new_node


    cdef search_level(self, BVECTOR query, hnswNode *entry_ptr, USHORT level):
        cdef DIST dist = hamming_dist(query, entry_ptr.vector)

        candidate_nodes = PriorityQueue()
        result_nodes = PriorityQueue()
        candidate_nodes.push((entry_ptr.id,), dist) # min priority queue
        result_nodes.push((entry_ptr.id,), 1/(dist+1)) # max priority queue

        cdef cpp_set[UIDX] visited_nodes

        cdef hnswNode *candidate
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

                if visited_nodes.find(id) != visited_nodes.end():
                    next_edge = next_edge.next
                    continue

                visited_nodes.insert(id)
                neighbor = self._get_node(id)

                dist = hamming_dist(query, neighbor.vector)

                # TODO: add hnswConfig
                if dist < lower_bound or result_nodes.size < 100:
                    candidate_nodes.push((id,), dist)
                    result_nodes.push((id,), 1/(dist+1.0))

                    if dist < lower_bound:
                        lower_bound = dist

                    if result_nodes.size > 100:
                       result_nodes.pop()

                next_edge = next_edge.next


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
                dist = hamming_dist(query, node_ptr.vector)
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
            candidates.push(item, 1 / priority - 0.1)
            neighbors_clone.push(item, priority)
            existing_candidates.insert(item[0])


        # TODO: add hnswConfig
        cdef USHORT candidates_size = neighbors.size()

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
                    if existing_candidates.find(id) != existing_candidates.end():
                        next_edge = next_edge.next
                        continue
                    existing_candidates.insert(id)
                    candidate = self._get_node(id)

                    dist = hamming_dist(query, candidate.vector)
                    candidates.push(item, dist)
        result = PriorityQueue()
        while (not candidates.empty()) and result.size < ensure_k:
            p, t = candidates.pop()
            result.push(t, p)

        return result


    cdef USHORT _random_level(self):
        cdef double r = rand() / RAND_MAX
        cdef double f = floor(-log(r) * 6)

        return int(f)

    cpdef query(self, BVECTOR query):
        #cdef array.array final_result = array.array('L')
        #cdef array.array final_idx = array.array('L')

        cdef hnswNode* entry_ptr = self.entry_ptr
        cdef DIST min_dist = hamming_dist(query, entry_ptr.vector)
        cdef USHORT l = self.max_level
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
                'vector': node.vector,
                'distance': dist
            })

        return result

    cpdef batch_query(self, BVECTOR query, const USHORT num_query):
        cdef UIDX _0
        cdef USHORT _1
        cdef BVECTOR q_key = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * self.bytes_per_vector)
        result = []
        for _0 in range(num_query):
            for _1 in range(self.bytes_per_vector):
                q_key[_1] = query[_1]
            q_result = self.query(q_key)
            result.append(q_result)

            query += self.bytes_per_vector

        PyMem_Free(q_key)

        return result

    cpdef void save_model(self, model_path):
        pass

    cpdef void load_model(self, model_path):
        pass

    @property
    def size(self):
        return self.total_size

    @property
    def memory_size(self):
        raise NotImplemented
        #return get_memory_size(self.root_node)

    def __cinit__(self, bytes_per_vector):
        self.bytes_per_vector = bytes_per_vector
        self.entry_ptr = NULL
        self.max_level = 0
        self.nodes_ptr = <prehash_map*> PyMem_Malloc(sizeof(prehash_map))

    def __dealloc__(self):
        # self.free_trie()
        pass

    cpdef destroy(self):
        pass
