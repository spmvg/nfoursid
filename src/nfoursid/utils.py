from collections import namedtuple
from typing import Tuple

import numpy as np

Decomposition = namedtuple('Decomposition', ['left_orthogonal', 'eigenvalues', 'right_orthogonal'])
"""
Eigenvalue decomposition of a matrix ``matrix`` such that ``left_orthogonal @ eigenvalues @ right_orthogonal``
equals ``matrix``.
"""


class Utils:
    @staticmethod
    def validate_matrix_shape(
            matrix: np.ndarray,
            shape: Tuple[float, float],
            name: str
    ):
        """
        Raises if ``matrix`` does not have shape ``shape``. The error message will contain ``name``.
        """
        if matrix.shape != shape:
            raise ValueError(f'Dimensions of `{name}` {matrix.shape} are inconsistent. Expected {shape}.')

    @staticmethod
    def eigenvalue_decomposition(
            matrix: np.ndarray
    ) -> Decomposition:
        """
        Calculate eigenvalue decomposition of ``matrix`` as a ``Decomposition``.
        """
        u, eigenvalues, vh = np.linalg.svd(matrix)
        eigenvalues_mat = np.zeros((u.shape[0], vh.shape[0]))
        np.fill_diagonal(eigenvalues_mat, eigenvalues)
        return Decomposition(u, eigenvalues_mat, vh)

    @staticmethod
    def reduce_decomposition(
            decomposition: Decomposition,
            rank: int
    ) -> Decomposition:
        """
        Reduce an eigenvalue decomposition ``decomposition`` such that only ``rank`` number of biggest eigenvalues
        remain. Returns another ``Decomposition``.
        """
        u, s, vh = decomposition
        return Decomposition(
            u[:, :rank],
            s[:rank, :rank],
            vh[:rank, :]
        )

    @staticmethod
    def block_hankel_matrix(
            matrix: np.ndarray,
            num_block_rows: int
    ) -> np.ndarray:
        """
        Calculate a block Hankel matrix based on input matrix ``matrix`` with ``num_block_rows`` block rows.
        The shape of ``matrix`` is interpreted in row-order, like the structure of a ``pd.DataFrame``:
        the rows are measurements and the columns are data sources.

        The returned block Hankel matrix has a columnar structure. Every column of the returned matrix consists
        of ``num_block_rows`` block rows (measurements). See the examples for details.

        Examples
        --------
        Suppose that the input matrix contains 4 measurements of 2-dimensional data:

        >>> matrix = np.array([
        >>>     [0, 1],
        >>>     [2, 3],
        >>>     [4, 5],
        >>>     [6, 7]
        >>> ])

        If the number of block rows is set to ``num_block_rows=2``, then the block Hankel matrix will be

        >>> np.array([
        >>>     [0, 2, 4],
        >>>     [1, 3, 5],
        >>>     [2, 4, 6],
        >>>     [3, 5, 7]
        >>> ])
        """
        hankel_rows_dim = num_block_rows * matrix.shape[1]
        hankel_cols_dim = matrix.shape[0] - num_block_rows + 1

        hankel = np.zeros((hankel_rows_dim, hankel_cols_dim))
        for block_row_index in range(hankel_cols_dim):
            flattened_block_rows = matrix[block_row_index:block_row_index+num_block_rows,
                                          :].flatten()
            hankel[:, block_row_index] = flattened_block_rows
        return hankel

    @staticmethod
    def vectorize(
            matrix: np.ndarray
    ) -> np.ndarray:
        """
        Given a matrix ``matrix`` of shape ``(a, b)``, return a vector of shape ``(a*b, 1)`` with all columns of
        ``matrix`` stacked on top of eachother.
        """
        return np.reshape(matrix.flatten(order='F'), (matrix.shape[0] * matrix.shape[1], 1))

    @staticmethod
    def unvectorize(
            vector: np.ndarray,
            num_rows: int
    ) -> np.ndarray:
        """
        Given a vector ``vector`` of shape ``(num_rows*b, 1)``, return a matrix of shape ``(num_rows, b)`` such that
        the stacked columns of the returned matrix equal ``vector``.
        """
        if vector.shape[0] % num_rows != 0 or vector.shape[1] != 1:
            raise ValueError(f'Vector shape {vector.shape} and `num_rows`={num_rows} are incompatible')
        return vector.reshape((num_rows, vector.shape[0] // num_rows), order='F')
