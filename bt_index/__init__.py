from collections import Counter

__version__ = '0.0.1'

__all__ = ['BTIndex']


class BTIndex:
    class Node:
        child = None
        left = None
        right = None
        key = None

    class Statistic:
        num_keys = 0
        num_unique_keys = 0
        max_len_list = 0

    def __init__(self, all_bytes, bytes_per_vector=96):
        self.root = self.Node()
        self.stat = self.Statistic()
        self.bytes_iter = iter(all_bytes)
        self.bytes_per_vector = bytes_per_vector
        self._add_all()

    def _add_all(self):
        while True:
            try:
                self._add_one()
                self.stat.num_keys += 1
                if self.stat.num_keys % 1000 == 0:
                    print(f'indexing {self.stat.num_keys}...')
            except StopIteration:
                break
                pass

    def _set_node_move_to_child(self, node, key):
        node.key = key
        if not node.child:
            node.child = self.Node()
        return node.child

    def _add_one(self):
        node = self.root
        for _ in range(self.bytes_per_vector):
            key = self.bytes_iter.__next__()
            while node:
                if not node.key or (node.key and key == node.key):
                    node = self._set_node_move_to_child(node, key)
                    break
                elif key < node.key:
                    if not node.left: node.left = self.Node()
                    node = node.left
                elif key > node.key:
                    if not node.right: node.right = self.Node()
                    node = node.right

        # all previous ops serve as key building for an inverted-index
        # now we are ready to indexing
        if node.key:
            node.key += 1
            if node.key > self.stat.max_len_list:
                self.stat.max_len_list = node.key
        else:
            node.key = 1
            self.stat.num_unique_keys += 1

    def get_all_list_len(self):
        all_len = []

        def preorder(node):
            if node:
                if not node.child:
                    all_len.append(node.key)
                preorder(node.child)
                preorder(node.left)
                preorder(node.right)

        preorder(self.root)
        assert len(all_len) == self.stat.num_unique_keys
        return Counter(all_len)

    def full_statistic(self):
        pass
