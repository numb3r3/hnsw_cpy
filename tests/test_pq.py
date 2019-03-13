import unittest

from hnsw_cpy.cython_core.heappq import PriorityQueue
import numpy as np
import random



class TestPriorityQueue(unittest.TestCase):
    def setUp(self):
        self.entries = [(("a", 123), 3),
                        (("b", [1, 2, 3]), 2),
                        (("c", "hello"), 6),
                        (("d", "world"), 4)]

        self.entries = [("x", x) for x in range(1, 10000)]
        random.shuffle(self.entries)

        self.pq = PriorityQueue()
        for item, weight in self.entries:
            self.pq.push(weight, item)


    def test_push(self):
        self.pq.push(0, ("e", 100))
        self.assertEqual(self.pq.size, 10000)

        weight, item = self.pq.pop_min()
        self.assertEqual(weight, 0)

        self.pq.push(10000, ("x"))
        self.assertEqual(self.pq.size, 10000)
        weight, item = self.pq.pop_max()
        self.assertEqual(weight, 10000)


    def test_pop_min(self):
        expects = list(range(1, 10000))
        i = 0
        while self.pq.size > 0:
            weight, item = self.pq.pop_min()
            self.assertEqual(weight, expects[i])
            i += 1


    def test_pop_max(self):
        expects = list(range(1, 10000))[::-1]
        i = 0
        while self.pq.size > 0:
            weight, item = self.pq.pop_max()
            self.assertEqual(weight, expects[i])
            i += 1


    def test_peak_min(self):
        while self.pq.size > 0:
            weight_1, item_1 = self.pq.peak_min()
            weight_2, item_2 = self.pq.pop_min()
            self.assertEqual(weight_1, weight_2)


    def test_peak_max(self):
        while self.pq.size > 0:
            weight_1, item_1 = self.pq.peak_max()
            weight_2, item_2 = self.pq.pop_max()
            self.assertEqual(weight_1, weight_2)

if __name__ == '__main__':
    unittest.main()
