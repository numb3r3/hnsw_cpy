# cython: language_level=3

from libc.stdlib cimport malloc, realloc, free

cdef unsigned int _BUCKET_CAPACITY

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

cdef _bucket* _new_bucket(unsigned int capacity, unsigned int offset):
    cdef _bucket* bucket = <_bucket*> malloc(sizeof(_bucket))
    bucket.data = <_value*> malloc(sizeof(_value)*capacity)
    bucket.capacity = capacity
    bucket.offset = offset
    bucket.size += 1

    return bucket

cdef void _bucket_insert(_bucket* bucket, unsinged int id, _value value)
    bucket.data[id] = value
    bucket.size += 1

cdef _value _bucket_get(_bucket* bucket, unsigned int id):
    return bucket.data[id]

cdef void _bucket_delete(_bucket* bucket, unsigned int id):
    bucket.data[id] = NULL
    # WIP: bucket.size is an unsigned integer
    if bucket.size > 0:
        bucket.size -= 1

cdef void _bucket_free(_bucket* bucket):
    free(bucket.data)
    free(bucket)

cdef prehash_map* new_prehash_map():
    cdef prehash_map* map = <prehash_map*> malloc(sizeof(prehash_map))
    map.total_size = 0

    map.buckets = <_bucket**> malloc(sizeof(_bucket*))
    map.buckets[0] = _new_bucket(BUCKET_CAPACITY, 0)
    map.bucket_count = 1

    return map


cdef void prehash_insert(prehash_map* map, unsigned int id, _value value):
     cdef unsigned int bucket_id = _get_bucket_id(id, BUCKET_CAPACITY)

     if bucket_id >= map.bucket_count:
         map.buckets = <_bucket**> realloc(map.buckets, sizeof(_bucket*) * (bucket_id+1))
         map.bucket_count = bucket_id+1

     if map.buckets[bucket_id] == NULL:
         map.buckets[bucket_id] = _new_bucket(BUCKET_CAPACITY, BUCKET_CAPACITY*bucket_id)

     cdef _bucket* bucket = map.buckets[bucket_id]

     id = id - BUCKET_CAPACITY*bucket_id

     _bucket_insert(bucket, id, value)
     map.total_size += 1

cdef _value prehash_get(prhash_map* map, unsigned int id):
    cdef unsigned int bucket_id = _get_bucket_id(id, BUCKET_CAPACITY)
    if bucket_id >= map.bucket_count:
        return NULL
    cdef _bucket* bucket = map.buckets[bucket_id]
    if bucket == NULL:
        return NULL
    id = id - BUCKET_CAPACITY*bucket_id
    return _bucket_get(bucket, id)

cdef void prehash_delete(prehash_map* map, unsigned int id):
    cdef unsigned int bucket_idx = _get_bucket_id(id, BUCKET_CAPACITY)
    if bucket_id >= map.bucket_count:
        return

    cdef _bucket* bucket = map.buckets[bucket_id]
    if bucket == NULL:
        return
    id = id - BUCKET_CAPACITY*bucket_id
    _bucket_delete(bucket, id)


cdef unsigned int _get_bucket_id(unsigned int id, unsigned int capacity):
    cdef unsigned int i = int(id / capacity)
    return i

cdef void prehash_free(prehash_map* hash_map):
    cdef unsigned int i
    for i in range(hash_map.bucket_size):
        _bucket_free(prehash_map.buckets[i])
    free(hash_map)
