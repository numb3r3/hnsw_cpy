import unittest

from hnsw_cpy.cython_lib.prehash import PrehashMap


class TestPriorityQueue(unittest.TestCase):
    def setUp(self):
        self.items = [
            (1, "a"),
            (2, "b"),
            (20010, "c"),
            (10005, 'd'),
            (31000, 'e')]

        self.hash_map = PrehashMap()
        for item in self.items:
            i, d = item
            self.hash_map.insert(i, d)


    def test_insert(self):
        self.assertEqual(self.hash_map.size, 5)

    def test_get(self):
        self.hash_map.insert(5, "f")
        self.hash_map.insert(40000, "g")
        self.assertEqual(self.hash_map.get(5), "f")
        self.assertEqual(self.hash_map.get(40000), "g")

    def test_delete(self):
        self.hash_map.delete(1)
        self.assertEqual(self.hash_map.size, 4)



if __name__ == '__main__':
    unittest.main()
