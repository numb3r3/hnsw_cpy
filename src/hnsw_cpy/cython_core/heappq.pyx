# cython: language_level=3

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef heappq* init_heappq():
    cdef heappq* pq = <heappq*> PyMem_Malloc(sizeof(heappq))
    pq.root = NULL
    pq.min_node = NULL
    pq.max_node = NULL
    pq.size = 0
    return pq

cdef void free_heappq(heappq* pq):
    cdef unsigned int i = 0
    while pq.size > 0:
        if i % 2 == 0:
            heappq_pop_max(pq)
        else:
            heappq_pop_min(pq)
    PyMem_Free(pq)

cdef void heappq_push(heappq* pq, float priority, value_t value):
    cdef pq_entity* entity = <pq_entity*> PyMem_Malloc(sizeof(pq_entity))
    entity.value = value
    entity.priority = priority

    cdef pq_node* new_node = <pq_node*> PyMem_Malloc(sizeof(pq_node))
    new_node.entity = entity
    new_node.parent = NULL
    new_node.left = NULL
    new_node.right = NULL

    pq.size += 1

    cdef pq_node* start_node = pq.root
    if start_node == NULL:
        pq.root = new_node
        pq.min_node = new_node
        pq.max_node = new_node
        return

    cdef bint new_min_node = 1
    cdef bint new_max_node = 1

    while start_node != NULL:
        if priority > start_node.entity.priority:
            new_min_node = 0
            if start_node.right != NULL:
                start_node = start_node.right
            else:
                new_node.parent = start_node
                start_node.right = new_node
                # pq.max_node = new_node
                break
        elif start_node.left != NULL:
            new_max_node = 0
            start_node = start_node.left
        else:
            new_max_node = 0
            new_node.parent = start_node
            start_node.left = new_node
            #pq.min_node = new_node
            break
    if new_min_node:
        pq.min_node = new_node
    elif new_max_node:
        pq.max_node = new_node


cdef pq_entity* heappq_pop_min(heappq* pq):
    if pq.min_node == NULL:
        return NULL

    pq.size -= 1

    cdef pq_node* result_node = pq.min_node
    cdef pq_node* parent_node = result_node.parent
    cdef pq_node* right_node = result_node.right

    if right_node == NULL:
        if parent_node != NULL:
           parent_node.left = NULL
        pq.min_node = parent_node
    else:
        if parent_node == NULL:
            pq.root = right_node

        result_node.right = NULL
        right_node.parent = parent_node
        if parent_node != NULL:
            parent_node.left = right_node

        pq.min_node = right_node
        while pq.min_node.left != NULL:
            pq.min_node = pq.min_node.left

    cdef pq_entity* result = result_node.entity
    result_node.entity = NULL
    PyMem_Free(result_node)


    return result


cdef pq_entity* heappq_pop_max(heappq* pq):
    if pq.max_node == NULL:
        return NULL

    pq.size -= 1

    cdef pq_node* result_node = pq.max_node
    cdef pq_node* parent_node = result_node.parent
    cdef pq_node* left_node =result_node.left

    if left_node == NULL:
        if parent_node != NULL:
            parent_node.right = NULL
        pq.max_node = parent_node
    else:
        if parent_node == NULL:
            pq.root = left_node

        result_node.left = NULL
        left_node.parent = parent_node
        if parent_node != NULL:
            parent_node.right = left_node

        pq.max_node = left_node
        while pq.max_node.right != NULL:
            pq.max_node = pq.max_node.right

    cdef pq_entity* result = result_node.entity

    result_node.entity = NULL
    PyMem_Free(result_node)

    return result


cdef pq_entity* heappq_peak_min(heappq* pq):
     if pq.min_node == NULL:
         return NULL
     return pq.min_node.entity

cdef pq_entity* heappq_peak_max(heappq* pq):
    if pq.max_node == NULL:
        return NULL
    return pq.max_node.entity


cdef class PriorityQueue(object):
    cdef heappq* pq

    def __init__(self):
        self.pq = init_heappq()

    def push(self, priority, item):
        heappq_push(self.pq, priority, <void*> item)

    def peak_min(self):
        cdef pq_entity* e = heappq_peak_min(self.pq)
        if e == NULL:
            return (None, None)
        return (e.priority, <object> e.value)

    def peak_max(self):
        cdef pq_entity* e = heappq_peak_max(self.pq)
        if e == NULL:
            return (None, None)
        return (e.priority, <object> e.value)

    def empty(self):
        return self.pq.size == 0

    def pop_min(self):
        cdef pq_entity* e = heappq_pop_min(self.pq)
        if e == NULL:
            return (None, None)
        return (e.priority, <object> e.value)

    def pop_max(self):
        cdef pq_entity* e = heappq_pop_max(self.pq)
        if e == NULL:
            return (None, None)
        return (e.priority, <object> e.value)

    @property
    def size(self):
        return self.pq.size

    def __dealloc__(self):
        free_heappq(self.pq)
