import unittest

import numpy as np

from nfoursid.utils import Utils


class TestUtils(unittest.TestCase):
    def test_block_hankel_matrix(self):
        matrix = np.array(range(15)).reshape((5, 3))
        hankel = Utils.block_hankel_matrix(matrix, 2)
        desired_result = np.array([
            [0., 3., 6., 9.],
            [1., 4., 7., 10.],
            [2., 5., 8., 11.],
            [3., 6., 9., 12.],
            [4., 7., 10., 13.],
            [5., 8., 11., 14.],
        ])
        self.assertTrue(np.all(np.isclose(desired_result, hankel)))

    def test_eigenvalue_decomposition(self):
        matrix = np.fliplr(np.diag(range(1, 3)))
        decomposition = Utils.eigenvalue_decomposition(matrix)
        self.assertTrue(np.all(np.isclose(
            [[0, -1],
             [-1, 0]],
            decomposition.left_orthogonal
        )))
        self.assertTrue(np.all(np.isclose(
            [2, 1],
            np.diagonal(decomposition.eigenvalues)
        )))
        self.assertTrue(np.all(np.isclose(
            [[-1, 0],
             [0, -1]],
            decomposition.right_orthogonal
        )))

        reduced_decomposition = Utils.reduce_decomposition(decomposition, 1)
        self.assertTrue(np.all(np.isclose(
            [[0], [-1]],
            reduced_decomposition.left_orthogonal
        )))
        self.assertTrue(np.all(np.isclose(
            [[2]],
            reduced_decomposition.eigenvalues
        )))
        self.assertTrue(np.all(np.isclose(
            [[-1, 0]],
            reduced_decomposition.right_orthogonal
        )))

    def test_vectorize(self):
        matrix = np.array([
            [0, 2],
            [1, 3]
        ])
        result = Utils.vectorize(matrix)
        self.assertTrue(np.all(np.isclose(
            np.array([
                [0],
                [1],
                [2],
                [3],
            ]),
            result
        )))

    def test_unvectorize(self):
        matrix = np.array([
            [0],
            [1],
            [2],
            [3],
        ])
        result = Utils.unvectorize(matrix, num_rows=2)
        self.assertTrue(np.all(np.isclose(
            np.array([
                [0, 2],
                [1, 3]
            ]),
            result
        )))

        with self.assertRaises(ValueError):
            Utils.unvectorize(matrix, num_rows=3)

        incompatible_matrix = matrix.T
        with self.assertRaises(ValueError):
            Utils.unvectorize(incompatible_matrix, 1)

    def test_validate_matrix_shape(self):
        with self.assertRaises(ValueError):
            Utils.validate_matrix_shape(
                np.array([[0]]),
                (42),
                'error'
            )
