# cython: language_level=3

from libc.stdlib cimport malloc, realloc, free
cimport cpython

cdef unsigned int BUCKET_CAPACITY = 10000

cdef _bucket* _new_bucket(unsigned int capacity, unsigned int offset):
    cdef _bucket* bucket = <_bucket*> malloc(sizeof(_bucket))
    cdef unsigned int _0
    bucket.data = <_value*> malloc(sizeof(_value)*capacity)
    for _0 in range(capacity):
        bucket.data[_0] = NULL
    bucket.capacity = capacity
    bucket.offset = offset
    bucket.size += 1

    return bucket

cdef void _bucket_insert(_bucket* bucket, unsigned int id, _value value):
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
    if bucket == NULL:
        return

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

     cdef unsigned int _0 = map.bucket_count

     if bucket_id >= map.bucket_count:
         map.buckets = <_bucket**> realloc(map.buckets, sizeof(_bucket*) * (bucket_id+1))
         map.bucket_count = bucket_id+1
         while _0 < map.bucket_count:
             map.buckets[_0] = _new_bucket(BUCKET_CAPACITY, BUCKET_CAPACITY*_0)
             _0 += 1

     cdef _bucket* bucket = map.buckets[bucket_id]

     id = id - BUCKET_CAPACITY*bucket_id

     _bucket_insert(bucket, id, value)
     map.total_size += 1

cdef _value prehash_get(prehash_map* map, unsigned int id):
    cdef unsigned int bucket_id = _get_bucket_id(id, BUCKET_CAPACITY)
    if bucket_id >= map.bucket_count:
        return NULL
    cdef _bucket* bucket = map.buckets[bucket_id]
    if bucket == NULL:
        return NULL
    id = id - BUCKET_CAPACITY*bucket_id
    return _bucket_get(bucket, id)

cdef void prehash_delete(prehash_map* map, unsigned int id):
    cdef unsigned int bucket_id = _get_bucket_id(id, BUCKET_CAPACITY)
    if bucket_id >= map.bucket_count:
        return

    cdef _bucket* bucket = map.buckets[bucket_id]
    if bucket == NULL:
        return
    id = id - BUCKET_CAPACITY*bucket_id
    _bucket_delete(bucket, id)
    map.total_size -= 1

cdef bint prehash_exist(prehash_map* map, unsigned int id):
    return prehash_get(map, id) != NULL


cdef bint prehash_is_empty(prehash_map* map):
    return map.total_size == 0


cdef unsigned int _get_bucket_id(unsigned int id, unsigned int capacity):
    cdef unsigned int i = int(id / capacity)
    return i

cdef void prehash_free(prehash_map* hash_map):
    cdef unsigned int i
    cdef _bucket* bucket
    for i in range(hash_map.bucket_count):
        _bucket_free(hash_map.buckets[i])
    free(hash_map)



cdef inline object fromvoidptr(void *a):
     cdef cpython.PyObject *o
     o = <cpython.PyObject *> a
     cpython.Py_XINCREF(o)
     return <object> o


cdef class PrehashMap(object):
    cdef prehash_map* _map_ptr

    def __cinit__(self):
        self._map_ptr = new_prehash_map()

    def insert(self, id, data):
        prehash_insert(self._map_ptr, id, <void*> data)

    def get(self, id):
        # return fromvoidptr(prehash_get(self._map_ptr, id))
        cdef _value value = prehash_get(self._map_ptr, id)
        if value == NULL:
            return None
        return <object> <void*> value

    def delete(self, id):
        prehash_delete(self._map_ptr, id)

    def is_empty(self):
        return prehash_is_empty(self._map_ptr)

    def exist(self, id):
        return prehash_exist(self._map_ptr, id)

    @property
    def size(self):
        return self._map_ptr.total_size

    def __dealloc__(self):
        prehash_free(self._map_ptr)
