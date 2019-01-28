# cython: language_level=3

cdef struct Node:
    Node*left
    Node*right
    Node*child
    unsigned char key
    unsigned long*value

cdef struct Statistic:
    unsigned long num_unique_keys
    unsigned long num_total_keys

cdef Node root
cdef unsigned short alloc_size_per_time = 100
cdef Statistic stat

cdef void _index_value(Node*node):
    print('here0')
    if node.value and node.value[0] == node.value[1]:
        new_value = <unsigned long*> PyMem_Realloc(node.value,
                                                   (node.value[1] + alloc_size_per_time) * sizeof(unsigned long))
        if not new_value:
            raise MemoryError()
        node.value = new_value
        node.value[1] += alloc_size_per_time
    elif not node.value:
        print('here1')
        node.value = <unsigned long*> PyMem_Malloc(alloc_size_per_time * sizeof(unsigned long))
        if not node.value:
            raise MemoryError()
        print('here2')
        node.value[0] = 0
        node.value[1] = alloc_size_per_time - 2  # first two are reserved for counting
        stat.num_unique_keys += 1
    (node.value + node.value[0] + 2)[0] = stat.num_total_keys
    node.value[0] += 1
    stat.num_total_keys += 1
    print('here3')

cdef Node*create_node():
    node = <Node*> PyMem_Malloc(sizeof(Node))
    node.left = NULL
    node.right = NULL
    node.child = NULL
    node.value = NULL
    node.key = 0
    return node

cpdef void index(unsigned char *data, const int bytes_per_vector, const int num_total):
    cdef Node*node
    for _0 in range(num_total):
        node = &root
        for _1 in range(bytes_per_vector):
            key = data[0]
            data = data + 1
            while node:
                if node.key == 0 or node.key == key:
                    node.key = key
                    if not node.child:
                        node.child = create_node()
                    node = node.child
                    break
                elif key < node.key:
                    if not node.left:
                        node.left = create_node()
                    node = node.left
                elif key > node.key:
                    if not node.right:
                        node.right = create_node()
                    node = node.right
        _index_value(node)
