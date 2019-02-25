import unittest

from bindex.cython_hnsw.utils import PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def setUp(self):
        self.entries = [(("a", 123), 3),
                        (("b", [1, 2, 3]), 2),
                        (("c", "hello"), 6),
                        (("d", "world"), 4)]

        self.pq = PriorityQueue()
        for item, weight in self.entries:
            self.pq.push(item, weight)


    def test_push(self):
        self.pq.push(("e", 100), 0)
        self.assertEqual(self.pq.size, 5)

        weight, item = self.pq.pop()
        self.assertEqual(weight, 0)
        self.assertEqual(item[0], "e")
        self.assertEqual(item[1], 100)


    def test_get(self):
        weight, item = self.pq.get("a")
        self.assertEqual(weight, 3)
        self.assertEqual(item[0], "a")
        self.assertEqual(item[1], 123)

    def test_pop(self):
        weight, item = self.pq.pop()
        self.assertEqual(weight, 2)
        self.assertEqual(item[0], "b")
        self.assertEqual(item[1], [1, 2, 3])

    def test_delete(self):
        self.pq.delete("d")
        self.assertEqual(self.pq.size, 3)
        entry = self.pq.get("d")
        self.assertEqual(entry, None)



if __name__ == '__main__':
    unittest.main()
