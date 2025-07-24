# This is a test based on the example Jupyter notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unittest

from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace

pd.set_option('display.max_columns', None)
np.random.seed(0)  # reproducable results

NUM_TRAINING_DATAPOINTS = 1000  # create a training-set by simulating a state-space model with this many datapoints
NUM_TEST_DATAPOINTS = 20  # same for the test-set
INPUT_DIM = 3
OUTPUT_DIM = 2
INTERNAL_STATE_DIM = 4  # actual order of the state-space model in the training- and test-set
NOISE_AMPLITUDE = .1  # add noise to the training- and test-set
FIGSIZE = 8
ORDER_OF_MODEL_TO_FIT = 4

# define system matrices for the state-space model of the training- and test-set
A = np.array([
    [1, .01, 0, 0],
    [0, 1, .01, 0],
    [0, 0, 1, .02],
    [0, -.01, 0, 1],
]) / 1.01
B = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
]
) / 3
C = np.array([
    [1, 0, 1, 1],
    [0, 0, 1, -1],
])
D = np.array([
    [1, 0, 1],
    [0, 1, 0]
]) / 10


class TestEndToEnd(unittest.TestCase):
    def test_end_to_end(self):
        state_space = StateSpace(A, B, C, D)
        for _ in range(NUM_TRAINING_DATAPOINTS):
            input_state = np.random.standard_normal((INPUT_DIM, 1))
            noise = np.random.standard_normal((OUTPUT_DIM, 1)) * NOISE_AMPLITUDE

            state_space.step(input_state, noise)

        nfoursid = NFourSID(
            state_space.to_dataframe(),  # the state-space model can summarize inputs and outputs as a dataframe
            output_columns=state_space.y_column_names,
            input_columns=state_space.u_column_names,
            num_block_rows=10
        )
        nfoursid.subspace_identification()

        state_space_identified, covariance_matrix = nfoursid.system_identification(
            rank=ORDER_OF_MODEL_TO_FIT
        )

        kalman = Kalman(state_space_identified, covariance_matrix)
        state_space = StateSpace(A, B, C, D)  # new data for the test-set
        for _ in range(NUM_TEST_DATAPOINTS):  # make a test-set
            input_state = np.random.standard_normal((INPUT_DIM, 1))
            noise = np.random.standard_normal((OUTPUT_DIM, 1)) * NOISE_AMPLITUDE

            y = state_space.step(input_state, noise)  # generate test-set
            kalman.step(y,
                        input_state)  # the Kalman filter sees the output and input, but not the actual internal state

        dataframe = kalman.to_dataframe()
        last_row = dataframe.iloc[-1, :].to_dict()

        expected_result = {
            ('$y_0$', 'actual', 'output'): -0.7076730559552813,
            ('$y_0$', 'filtered', 'output'): -0.8661379329447575,
            ('$y_0$', 'filtered', 'standard deviation'): 0.12009260340845236,
            ('$y_0$', 'next predicted (input corrected)', 'output'): np.nan,
            ('$y_0$', 'next predicted (input corrected)', 'standard deviation'): 0.12049142178612798,
            ('$y_0$', 'next predicted (no input)', 'output'): -1.332263284464381,
            ('$y_0$', 'next predicted (no input)', 'standard deviation'): 0.12049142178612798,
            ('$y_1$', 'actual', 'output'): -0.84406136346813,
            ('$y_1$', 'filtered', 'output'): -0.7894290554207276,
            ('$y_1$', 'filtered', 'standard deviation'): 0.11920775718600453,
            ('$y_1$', 'next predicted (input corrected)', 'output'): np.nan,
            ('$y_1$', 'next predicted (input corrected)', 'standard deviation'): 0.11962978819874831,
            ('$y_1$', 'next predicted (no input)', 'output'): -0.08334764426935866,
            ('$y_1$', 'next predicted (no input)', 'standard deviation'): 0.11962978819874831
        }
        for key, value in expected_result.items():
            if pd.notna(value):
                self.assertAlmostEqual(
                    value,
                    last_row[key],
                )
                continue
            self.assertTrue(
                pd.isnull(last_row[key]),
            )
