import unittest

from hnsw_cpy.cython_core.hnsw import hamming_dist


class TestHammingDist(unittest.TestCase):

    def test_hamming(self):
        x = bytes([1, 2, 3, 4])
        y = bytes([1, 2, 3, 5])
        z = bytes([2, 3, 5, 1])
        dist0 = hamming_dist(x, x, 4)
        dist1 = hamming_dist(x, y, 4)
        dist2 = hamming_dist(x, z, 4)

        self.assertEqual(dist0, 0)
        self.assertEqual(dist1, 1)
        self.assertEqual(dist2, 4)


if __name__ == '__main__':
    unittest.main()
