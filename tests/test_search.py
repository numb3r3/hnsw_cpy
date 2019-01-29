import unittest

from bindex import BIndex


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.index_data = {'data': bytes([1, 2, 3, 4,
                                          1, 2, 3, 5,
                                          1, 2, 3, 2,
                                          1, 2, 3, 4,
                                          5, 1, 2, 3,
                                          5, 1, 2, 4,
                                          5, 1, 2, 3,
                                          1, 1, 1, 1]),
                           'bytes': 4,
                           'num_keys': 8,
                           'num_unique_keys': 6}

        self.query_data = {'data': bytes([1, 2, 3, 4,
                                          6, 1, 2, 3,
                                          5, 1, 2, 3,
                                          1, 2, 3, 4,
                                          1, 2, 3, 5,
                                          6, 1, 2, 3]),
                           'expect': [True, False, True, True, True, False]}

    @staticmethod
    def _test_toy_data(index_data, query_data, store_mode):
        bt = BIndex(index_data['bytes'], store_mode)
        bt.add(index_data['data'])
        return bt.contains(query_data['data'])

    def test_basic(self):
        self.assertEqual(self.query_data['expect'],
                         self._test_toy_data(self.index_data, self.query_data, 'none'))

    def test_trie(self):
        self.assertEqual(self.query_data['expect'],
                         self._test_toy_data(self.index_data, self.query_data, 'trie'))


if __name__ == '__main__':
    unittest.main()
