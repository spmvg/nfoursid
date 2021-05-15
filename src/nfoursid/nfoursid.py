from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfoursid.state_space import StateSpace
from nfoursid.utils import Utils, Decomposition


class NFourSID:
    r"""
    Perform subspace identification using N4SID [1].
    The state-space model under consideration, is:

    .. math::
        \begin{cases}
            x_{k+1} &= A x_k + B u_k + K e_k \\
            y_k &= C x_k + D u_k + e_k
        \end{cases}

    Data is provided as a dataframe ``dataframe`` where every row is a measurement.
    The output columns are given by ``output_columns``.
    The input columns are given by ``input_columns``.

    The number of block rows to use in the block Hankel matrices is given by ``num_block_rows``.
    If ``num_block_rows`` is chosen to be too big, the computational complexity will increase.
    If ``num_block_rows`` is chosen to be too small, the order of the system might not be possible to determine
    in the eigenvalue diagram. Moreover, if ``num_block_rows`` is chosen to be too small,
    the assumptions of [2] might not hold.

    [1] Van Overschee, Peter, and Bart De Moor. "N4SID: Subspace algorithms for the identification of combined
    deterministic-stochastic systems." Automatica 30.1 (1994): 75-93.
    """
    def __init__(
            self,
            dataframe: pd.DataFrame,
            output_columns: List[str],
            input_columns: List[str] = None,
            num_block_rows: int = 2
    ):
        self.u_columns = input_columns or []
        self.y_columns = output_columns
        self.num_block_rows = num_block_rows

        self._set_input_output_data(dataframe)
        self._initialize_instance_variables()

    def _initialize_instance_variables(self):
        """ Initialize variables. """
        self.R22, self.R32 = None, None
        self.R32_decomposition = None
        self.x_dim = None

    def _set_input_output_data(
            self,
            dataframe: pd.DataFrame
    ):
        """ Perform data consistency checks and set timeseries data arrays. """
        u_frame = dataframe[self.u_columns]
        if u_frame.isnull().any().any():
            raise ValueError('Input data cannot contain nulls')
        y_frame = dataframe[self.y_columns]
        if y_frame.isnull().any().any():
            raise ValueError('Output data cannot contain nulls')
        self.u_array = u_frame.to_numpy()
        self.y_array = y_frame.to_numpy()
        self.u_dim = self.u_array.shape[1]
        self.y_dim = self.y_array.shape[1]

    def subspace_identification(self):
        """
        Perform subspace identification based on the PO-MOESP method.
        The instrumental variable contains past outputs and past inputs.
        The implementation uses a QR-decomposition for numerical efficiency and is based on page 329 of [1].

        A key result of this function is the eigenvalue decomposition of the :math:`R_{32}` matrix
        ``self.R32_decomposition``, based on which the order of the system should be determined.

        [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
        Cambridge university press, 2007.
        """
        u_hankel = Utils.block_hankel_matrix(self.u_array, self.num_block_rows)
        y_hankel = Utils.block_hankel_matrix(self.y_array, self.num_block_rows)

        u_past, u_future = u_hankel[:, :-self.num_block_rows], u_hankel[:, self.num_block_rows:]
        y_past, y_future = y_hankel[:, :-self.num_block_rows], y_hankel[:, self.num_block_rows:]
        u_instrumental_y = np.concatenate([u_future, u_past, y_past, y_future])

        q, r = map(lambda matrix: matrix.T, np.linalg.qr(u_instrumental_y.T, mode='reduced'))

        y_rows, u_rows = self.y_dim * self.num_block_rows, self.u_dim * self.num_block_rows
        self.R32 = r[-y_rows:, u_rows:-y_rows]
        self.R22 = r[u_rows:-y_rows, u_rows:-y_rows]
        self.R32_decomposition = Utils.eigenvalue_decomposition(self.R32)

    def system_identification(
            self,
            rank: int = None
    ) -> Tuple[StateSpace, np.ndarray]:
        """
        Identify the system matrices of the state-space model given in the description of ``NFourSID``.
        Moreover, the covariance of the measurement-noise and process-noise will be estimated.
        The order of the returned state-space model has rank ``rank`` by reducing the eigenvalue decomposition.
        The implementation is based on page 333 of [1].

        The return value consists of a tuple containing

        - The identified state-space model containing the estimated matrices :math:`(A, B, C, D)`,
        - and an estimate of the covariance matrix of the measurement-noise :math:`w`
          and process-noise :math:`v`.
          The structure of the covariance matrix corresponds to the parameter ``noise_covariance`` of
          ``subspace_identification.kalman.Kalman``.
          See its documentation for more information.

        ``self.system_identification`` needs the QR-decomposition result of subspace identification
        ``self.R32``, and therefore can only be ran after ``self.subspace_identification``.

        [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
        Cambridge university press, 2007.
        """
        if self.R32_decomposition is None:
            raise Exception('Perform subspace identification first.')
        if rank is None:
            rank = self.y_dim * self.num_block_rows
        self.x_dim = rank

        observability_decomposition = self._get_observability_matrix_decomposition()

        return self._identify_state_space(observability_decomposition)

    def _identify_state_space(
            self,
            observability_decomposition: Decomposition
    ) -> Tuple[StateSpace, np.ndarray]:
        """
        Approximate the row space of the state sequence of a Kalman filter as per the N4SID scheme.
        Then, solve a least squares problem to identify the system matrices.
        Finally, use the residuals to estimate the noise covariance matrix.
        """
        x = (np.power(observability_decomposition.eigenvalues, .5)
             @ observability_decomposition.right_orthogonal)[:, :-1]
        last_y, last_u = self.y_array[self.num_block_rows:, :].T, self.u_array[self.num_block_rows:, :].T
        x_and_y = np.concatenate([x[:, 1:],
                                  last_y[:, :-1]])
        x_and_u = np.concatenate([x[:, :-1],
                                  last_u[:, :-1]])
        abcd = (np.linalg.pinv(x_and_u @ x_and_u.T) @ x_and_u @ x_and_y.T).T
        residuals = x_and_y - abcd @ x_and_u
        covariance_matrix = residuals @ residuals.T / residuals.shape[1]
        q = covariance_matrix[:self.x_dim, :self.x_dim]
        r = covariance_matrix[self.x_dim:, self.x_dim:]
        s = covariance_matrix[:self.x_dim, self.x_dim:]
        state_space_covariance_matrix = np.concatenate(
            [
                np.concatenate([r, s.T], axis=1),
                np.concatenate([s, q], axis=1)
            ],
            axis=0
        )
        return (
            StateSpace(
                abcd[:self.x_dim, :self.x_dim],
                abcd[:self.x_dim, self.x_dim:],
                abcd[self.x_dim:, :self.x_dim],
                abcd[self.x_dim:, self.x_dim:],
            ),
            (state_space_covariance_matrix + state_space_covariance_matrix.T) / 2
        )

    def _get_observability_matrix_decomposition(self) -> Decomposition:
        """
        Calculate the eigenvalue decomposition of the estimate of the observability matrix as per N4SID.
        """
        u_hankel = Utils.block_hankel_matrix(self.u_array, self.num_block_rows)
        y_hankel = Utils.block_hankel_matrix(self.y_array, self.num_block_rows)
        u_and_y = np.concatenate([u_hankel, y_hankel])
        observability = self.R32 @ np.linalg.pinv(self.R22) @ u_and_y
        observability_decomposition = Utils.reduce_decomposition(
            Utils.eigenvalue_decomposition(observability),
            self.x_dim
        )
        return observability_decomposition

    def plot_eigenvalues(self, ax: plt.axes):  # pragma: no cover
        """
        Plot the eigenvalues of the :math:`R_{32}` matrix, so that the order of the state-space model can be determined.
        Since the :math:`R_{32}` matrix should have been calculated, this function can only be used after
        performing ``self.subspace_identification``.
        """
        if self.R32_decomposition is None:
            raise Exception('Perform subspace identification first.')

        ax.semilogy(np.diagonal(self.R32_decomposition.eigenvalues), 'x')
        ax.set_title('Estimated observability matrix decomposition')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.grid()
