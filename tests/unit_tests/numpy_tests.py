import unittest
import numpy as np

class TestNumpy(unittest.TestCase):
    def setUp(self):
        self.mat_2_2 = np.matrix('1 2; 3 4')

    def test_rank(self):
        rank = np.linalg.matrix_rank(self.mat_2_2)
        self.assertEqual(rank, 2)

    def test_create_identity_matrix(self):
        np.eye(5)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
