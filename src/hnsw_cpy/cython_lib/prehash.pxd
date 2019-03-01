# cython: language_level=3

from libc.stdlib cimport malloc, realloc, free
cimport cpython

ctypedef void* _value

cdef struct _bucket:
    _value* data
    unsigned int capacity
    unsigned int offset
    unsigned int size


cdef struct prehash_map:
    _bucket** buckets
    unsigned int bucket_count
    unsigned int total_size

cdef _bucket* _new_bucket(unsigned int capacity, unsigned int offset)

cdef void _bucket_insert(_bucket* bucket, unsigned int id, _value value)

cdef _value _bucket_get(_bucket* bucket, unsigned int id)

cdef void _bucket_delete(_bucket* bucket, unsigned int id)

cdef void _bucket_free(_bucket* bucket)

cdef prehash_map* new_prehash_map()

cdef void prehash_insert(prehash_map* map, unsigned int id, _value value)

cdef _value prehash_get(prehash_map* map, unsigned int id)

cdef void prehash_delete(prehash_map* map, unsigned int id)

cdef void prehash_free(prehash_map* hash_map)
