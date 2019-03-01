# cython: language_level=3

from libc.stdlib cimport malloc, free

ctypedef void *queue_value

cdef struct queue_entry:
	queue_value data
	queue_entry *prev
	queue_entry *next

cdef struct queue:
    queue_entry *head
	queue_entry *tail


cdef queue* new_queue():
    cdef queue* q = <queue*> malloc(sizeof(queue))
    q.head = NULL
    q.tail = NULL
    return q

cdef void queue_free(queue* q_ptr):
    while not queue_is_empty(q_ptr):
        queue_pop_head(q_ptr)

    free(q_ptr)

cdef void queue_push_head(queue *q_ptr, queue_value data):
    cdef queue_entry* entry = <queue_entry*> malloc(sizeof(queue_entry))
    entry.data = data
    entry.prev = NULL
    entry.next = q_ptr.head

    if q_ptr.head == NULL:
        q_ptr.head = entry
		q_ptr.tail = entry
    else:
        q_ptr.head.prev = entry
	    q_ptr.head = entry


cdef queue_value queue_pop_head(queue *q_ptr):
    cdef queue_entry *entry
	cdef queue_value result

	if queue_is_empty(q_ptr)
	    return NULL

	entry = q_ptr.head
	q_ptr.head = entry.next
	result = entry.data

    if q_ptr.head == NULL:
        q_ptr.tail = NULL
    else:
        q_ptr.head.prev = NULL

    free(entry)

	return result


cdef queue_value queue_peek_head(queue *q_ptr):
     if queue_is_empty(q_ptr):
         return NULL
     else:
         return q_ptr.head.data


cdef void queue_push_tail(queue *q_ptr, queue_value data):
    cdef queue_entry* entry = <queue_entry*> malloc(sizeof(queue_entry))
    entry.data = data
    entry.prev = NULL
    entry.next = NULL

    if queue.tail == NULL:
        q_ptr.head = entry
        q_ptr.tail = entry
    else:
        q_ptr.tail.next = entry
        q_ptr.tail = entry

cdef queue_value queue_pop_tail(queue* q_ptr):
    cdef queue_entry *entry
    cdef queue_value result

    if queue_is_empty(q_ptr):
        return NULL

    entry = q_ptr.tail
    q_ptr.tail = entry.prev
    result = entry.data

    if q_ptr.tail == NULL:
        q_ptr.head = NULL
    else:
        q_ptr.tail.next = NULL

    free(entry)
    return result

cdef queue_value queue_peek_tail(queue* q_ptr):
    if queue_is_empty(q_ptr):
        return NULL
    else:
        return q_ptr.tail.data

cdef queue_is_empty(queue* q_ptr):
    return q_ptr.head == NULL
