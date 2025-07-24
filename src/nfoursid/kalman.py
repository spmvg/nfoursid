from typing import List, Optional

from matplotlib import figure as matplotlib_figure
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nfoursid.state_space import StateSpace
from nfoursid.utils import Utils


class Kalman:
    r"""
    Implementation [1] of a Kalman filter for a state-space model ``state_space``:

    .. math::
        \begin{cases}
            x_{k+1} &= A x_k + B u_k + w_k \\
            y_k &= C x_k + D u_k + v_k
        \end{cases}

    The matrices :math:`(A, B, C, D)` are taken from the state-space model ``state_space``.
    The measurement-noise :math:`v_k` and process-noise :math:`w_k` have a covariance matrix
    ``noise_covariance`` defined as

    .. math::
        \texttt{noise\_covariance} := \mathbb{E} \bigg (
        \begin{bmatrix}
            v \\ w
        \end{bmatrix}
        \begin{bmatrix}
            v \\ w
        \end{bmatrix}^\mathrm{T}
        \bigg )

    [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
    Cambridge university press, 2007.
    """
    output_label = 'output'
    """ Label given to an output column in ``self.to_dataframe``. """
    standard_deviation_label = 'standard deviation'
    """ Label given to a standard deviation column in ``self.to_dataframe``. """
    actual_label = 'actual'
    """ Label given to a column in ``self.to_dataframe``, indicating measured values. """
    filtered_label = 'filtered'
    """ Label given to a column in ``self.to_dataframe``, indicating the filtered state of the Kalman filter. """
    next_predicted_label = 'next predicted (no input)'
    """
    Label given to a column in ``self.to_dataframe``, indicating the predicted state of the Kalman filter under the
    absence of further inputs.
    """
    next_predicted_corrected_label = 'next predicted (input corrected)'
    """
    Label given to a column in ``self.to_dataframe``, indicating the predicted state of the Kalman filter corrected
    by previous inputs. The inputs to the state-space model are known, but not at the time that the prediction was
    made. In order to make a fair comparison for prediction performance, the direct effect of the input on the output
    by the matrix :math:`D` is removed in this column.
    
    The latest prediction will have ``np.nan`` in this column, since the input is not yet known.
    """

    def __init__(
            self,
            state_space: StateSpace,
            noise_covariance: np.ndarray
    ):
        self.state_space = state_space

        Utils.validate_matrix_shape(
            noise_covariance,
            (self.state_space.y_dim + self.state_space.x_dim,
             self.state_space.y_dim + self.state_space.x_dim),
            'noise_covariance')
        self.r = noise_covariance[:self.state_space.y_dim, :self.state_space.y_dim]
        self.s = noise_covariance[self.state_space.y_dim:, :self.state_space.y_dim]
        self.q = noise_covariance[self.state_space.y_dim:, self.state_space.y_dim:]

        self.x_filtereds = []
        self.x_predicteds = []
        self.p_filtereds = []
        self.p_predicteds = []
        self.us = []
        self.ys = []
        self.y_filtereds = []
        self.y_predicteds = []
        self.kalman_gains = []

    def step(
            self,
            y: Optional[np.ndarray],
            u: np.ndarray
    ):
        """
        Given an observed input ``u`` and output ``y``, update the filtered and predicted states of the Kalman filter.
        Follows the implementation of the conventional Kalman filter in [1] on page 140.

        The output ``y`` can be missing by setting ``y=None``.
        In that case, the Kalman filter will obtain the next internal state by stepping the state space model.

        [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
        Cambridge university press, 2007.
        """
        if y is not None:
            Utils.validate_matrix_shape(y, (self.state_space.y_dim, 1), 'y')
        Utils.validate_matrix_shape(u, (self.state_space.u_dim, 1), 'u')

        x_pred = self.x_predicteds[-1] if self.x_predicteds else np.zeros((self.state_space.x_dim, 1))
        p_pred = self.p_predicteds[-1] if self.p_predicteds else np.eye(self.state_space.x_dim)

        k_filtered = p_pred @ self.state_space.c.T @ np.linalg.pinv(
            self.r + self.state_space.c @ p_pred @ self.state_space.c.T
        )

        self.p_filtereds.append(
            p_pred - k_filtered @ self.state_space.c @ p_pred
        )

        self.x_filtereds.append(
            x_pred + k_filtered @ (y - self.state_space.d @ u - self.state_space.c @ x_pred)
            if y is not None else x_pred
        )

        k_pred = (self.s + self.state_space.a @ p_pred @ self.state_space.c.T) @ np.linalg.pinv(
            self.r + self.state_space.c @ p_pred @ self.state_space.c.T
        )

        self.p_predicteds.append(
            self.state_space.a @ p_pred @ self.state_space.a.T
            + self.q
            - k_pred @ (self.s + self.state_space.a @ p_pred @ self.state_space.c.T).T
        )

        x_predicted = self.state_space.a @ x_pred + self.state_space.b @ u
        if y is not None:
            x_predicted += k_pred @ (y - self.state_space.d @ u - self.state_space.c @ x_pred)
        self.x_predicteds.append(
            x_predicted
        )

        self.us.append(u)
        self.ys.append(y if y is not None else np.full((self.state_space.y_dim, 1), np.nan))
        self.y_filtereds.append(self.state_space.output(self.x_filtereds[-1], self.us[-1]))
        self.y_predicteds.append(self.state_space.output(self.x_predicteds[-1]))
        self.kalman_gains.append(k_pred)

        return self.y_filtereds[-1], self.y_predicteds[-1]

    def extrapolate(
            self,
            timesteps
    ) -> pd.DataFrame:
        """
        Make a ``timesteps`` number of steps ahead prediction about the output of the state-space model
        ``self.state_space`` given no further inputs.
        The result is a ``pd.DataFrame`` where the columns are ``self.state_space.y_column_names``:
        the output columns of the state-space model ``self.state_space``.
        """
        if not self.x_predicteds:
            raise Exception('Prediction is only possible once Kalman estimation has been performed.')

        state_space = StateSpace(
            self.state_space.a,
            self.state_space.b,
            self.state_space.c,
            self.state_space.d,
            x_init=self.x_predicteds[-1],
            y_column_names=self.state_space.y_column_names,
            u_column_names=self.state_space.u_column_names
        )

        for _ in range(timesteps):
            state_space.step()

        return state_space.to_dataframe()[state_space.y_column_names]

    def _measurement_and_state_standard_deviation(
            self,
            state_covariance_matrices: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Calculates the expected standard deviations on the output, assuming independence (!) in the noise of the state
        estimate and the process noise.
        Returns a list of row-vectors containing the standard deviations for the outputs.
        """
        covars_process_y = [
            self.state_space.c @ p @ self.state_space.c.T for p in state_covariance_matrices
        ]

        var_process_ys = [
            np.maximum(
                np.diagonal(p), 0
            )
            for p in covars_process_y
        ]
        var_measurement_y = np.maximum(np.diagonal(self.r), 0)

        return [
            np.sqrt(
                var_process_y + var_measurement_y
            ).reshape(
                (self.state_space.y_dim, 1)
            )
            for var_process_y in var_process_ys
        ]

    @staticmethod
    def _list_of_states_to_array(
            list_of_states: List[np.ndarray]
    ) -> np.ndarray:
        return np.array(list_of_states).squeeze(axis=2)

    @staticmethod
    def _reduce_dimension(element):
        return element[0]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the output of the Kalman filter as a ``pd.DataFrame``. The returned value contains information about
        filtered and predicted states of the Kalman filter at different timesteps.
        The expected standard deviation of the output is given, assuming independence (!) of the state estimation error
        and measurement noise.

        The rows of the returned dataframe correspond to timesteps.
        The columns of the returned dataframe are a 3-dimensional multi-index with the following levels:

        1. The output name, in the list ``self.state_space.y_column_names``.
        2. An indication of whether the value is
            - a value that was actually measured, these values were given to `self.step` as the `y` parameter,
            - a filtered state,
            - a predicted state given no further input or
            - a predicted state where the effect of the next input has been corrected for.
              This column is useful for comparing prediction performance.
        3. Whether the column is a value or the corresponding expected standard deviation.
        """
        input_corrected_predictions = [
            output + self.state_space.d @ input_state
            for input_state, output
            in zip(self.us[1:], self.y_predicteds[:-1])
        ] + [np.empty((self.state_space.y_dim, 1)) * np.nan]

        output_frames = [
            pd.DataFrame({
                (self.actual_label, self.output_label): map(self._reduce_dimension, outputs),
                (self.filtered_label, self.output_label): map(self._reduce_dimension, filtereds),
                (self.filtered_label, self.standard_deviation_label): map(self._reduce_dimension, filtered_stds),
                (self.next_predicted_label, self.output_label): map(self._reduce_dimension, predicteds),
                (self.next_predicted_label, self.standard_deviation_label): map(self._reduce_dimension, predicted_stds),
                (self.next_predicted_corrected_label, self.output_label): map(self._reduce_dimension, input_corrected_prediction),
                (self.next_predicted_corrected_label, self.standard_deviation_label): map(self._reduce_dimension, predicted_stds),
            })
            for (
                outputs,
                filtereds,
                predicteds,
                filtered_stds,
                predicted_stds,
                input_corrected_prediction
            ) in zip(
                zip(*self.ys),
                zip(*self.y_filtereds),
                zip(*self.y_predicteds),
                zip(*self._measurement_and_state_standard_deviation(self.p_filtereds)),
                zip(*self._measurement_and_state_standard_deviation(self.p_predicteds)),
                zip(*input_corrected_predictions)
            )
        ]
        return pd.concat(output_frames, axis=1, keys=self.state_space.y_column_names)

    def plot_filtered(self, fig: matplotlib_figure.Figure):  # pragma: no cover
        """
        The top graph plots the filtered output states of the Kalman filter and compares with the measured values.
        The error bars correspond to the expected standard deviations.
        The bottom graph zooms in on the errors between the filtered states and the measured values, compared with
        the expected standard deviations.
        """
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        df = self.to_dataframe()

        top_legends, bottom_legends = [], []
        for output_name in self.state_space.y_column_names:
            actual_outputs = df[(output_name, self.actual_label, self.output_label)]
            predicted_outputs = df[(output_name, self.filtered_label, self.output_label)]
            std = df[(output_name, self.filtered_label, self.standard_deviation_label)]

            markers, = ax1.plot(
                list(range(len(actual_outputs))),
                actual_outputs,
                'x'
            )
            line, = ax1.plot(
                list(range(len(actual_outputs))),
                actual_outputs,
                '-',
                color=markers.get_color(),
                alpha=.15
            )
            top_legends.append(((markers, line), output_name))

            prediction_errorbar = ax1.errorbar(
                list(range(len(predicted_outputs))),
                predicted_outputs,
                yerr=std,
                marker='_',
                alpha=.5,
                color=markers.get_color(),
                markersize=10,
                linestyle='',
                capsize=3
            )
            top_legends.append((prediction_errorbar, f'Filtered {output_name}'))

            errors = actual_outputs - predicted_outputs
            markers_bottom, = ax2.plot(
                list(range(len(errors))),
                errors,
                'x'
            )
            lines_bottom, = ax2.plot(
                list(range(len(errors))),
                errors,
                '-',
                color=markers_bottom.get_color(),
                alpha=.15
            )
            bottom_legends.append(((markers_bottom, lines_bottom), f'Error {output_name}'))
            prediction_errorbar_bottom, = ax2.plot(
                list(range(len(predicted_outputs))),
                std,
                '--',
                alpha=.5,
                color=markers.get_color(),
            )
            ax2.plot(
                list(range(len(predicted_outputs))),
                -std,
                '--',
                alpha=.5,
                color=markers.get_color(),
            )
            bottom_legends.append((prediction_errorbar_bottom, rf'Filtered $\sigma(${output_name}$)$'))

        lines, names = zip(*top_legends)
        ax1.legend(lines, names, loc='upper left')
        ax1.set_ylabel('Output $y$ (a.u.)')
        ax1.grid()

        lines, names = zip(*bottom_legends)
        ax2.legend(lines, names, loc='upper left')
        ax2.set_xlabel('Index')
        ax2.set_ylabel(r'Filtering error $y-y_{\mathrm{filtered}}$ (a.u.)')
        ax2.grid()
        ax1.set_title('Kalman filter, filtered state')
        plt.setp(ax1.get_xticklabels(), visible=False)

    def plot_predicted(
            self,
            fig: matplotlib_figure.Figure,
            steps_to_extrapolate: int = 1
    ):  # pragma: no cover
        """
        The top graph plots the predicted output states of the Kalman filter and compares with the measured values.
        The error bars correspond to the expected standard deviations.

        The stars on the top right represent the ``steps_to_extrapolate``-steps ahead extrapolation under no further
        inputs. The bottom graph zooms in on the errors between the predicted states and the measured values, compared
        with the expected standard deviations.
        """
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        df = self.to_dataframe()

        extrapolation = self.extrapolate(steps_to_extrapolate)

        top_legends, bottom_legends = [], []
        for output_name in self.state_space.y_column_names:
            actual_outputs = df[(output_name, self.actual_label, self.output_label)]
            predicted_outputs = df[(output_name, self.next_predicted_corrected_label, self.output_label)]
            std = df[(output_name, self.next_predicted_label, self.standard_deviation_label)]
            last_predicted_std = std.iloc[-1]

            markers, = ax1.plot(
                list(range(len(actual_outputs))),
                actual_outputs,
                'x'
            )
            line, = ax1.plot(
                list(range(len(actual_outputs))),
                actual_outputs,
                '-',
                color=markers.get_color(),
                alpha=.15
            )
            top_legends.append(((markers, line), output_name))

            prediction_errorbar = ax1.errorbar(
                list(range(1, len(predicted_outputs)+1)),
                predicted_outputs,
                yerr=std,
                marker='_',
                alpha=.5,
                color=markers.get_color(),
                markersize=10,
                linestyle='',
                capsize=3
            )
            top_legends.append((prediction_errorbar, f'Predicted {output_name}'))
            extrapolation_errorbar = ax1.errorbar(
                list(range(len(self.ys), len(self.ys) + steps_to_extrapolate)),
                extrapolation[output_name].to_numpy(),
                yerr=last_predicted_std,
                marker='*',
                markersize=9,
                alpha=.8,
                color=markers.get_color(),
                linestyle='',
                capsize=3
            )
            top_legends.append((extrapolation_errorbar, f'Extrapolation {output_name} (no input)'))

            errors = actual_outputs.to_numpy()[1:] - predicted_outputs.to_numpy()[:-1]
            markers_bottom, = ax2.plot(
                list(range(1, len(errors)+1)),
                errors,
                'x'
            )
            lines_bottom, = ax2.plot(
                list(range(1, len(errors)+1)),
                errors,
                '-',
                color=markers_bottom.get_color(),
                alpha=.15
            )
            bottom_legends.append(((markers_bottom, lines_bottom), f'Error {output_name}'))
            prediction_errorbar_bottom, = ax2.plot(
                list(range(1, len(predicted_outputs))),
                std[:-1],
                '--',
                alpha=.5,
                color=markers.get_color(),
            )
            ax2.plot(
                list(range(1, len(predicted_outputs))),
                -std[:-1],
                '--',
                alpha=.5,
                color=markers.get_color(),
            )
            bottom_legends.append((prediction_errorbar_bottom, rf'Predicted $\sigma(${output_name}$)$'))

        lines, names = zip(*top_legends)
        ax1.legend(lines, names, loc='upper left')
        ax1.set_ylabel('Output $y$ (a.u.)')
        ax1.grid()

        lines, names = zip(*bottom_legends)
        ax2.legend(lines, names, loc='upper left')
        ax2.set_xlabel('Index')
        ax2.set_ylabel(r'Prediction error $y-y_{\mathrm{predicted}}$ (a.u.)')
        ax2.grid()
        ax1.set_title('Kalman filter, predicted state')
        plt.setp(ax1.get_xticklabels(), visible=False)