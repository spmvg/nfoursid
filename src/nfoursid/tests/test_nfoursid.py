import unittest

import numpy as np
import pandas as pd

from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace


class TestNFourSid(unittest.TestCase):
    def test_which_is_actually_regression_test(self):
        n_datapoints = 100
        model = StateSpace(
            np.array([[.5]]),
            np.array([[.6]]),
            np.array([[.7]]),
            np.array([[.8]]),
        )
        np.random.seed(0)
        for _ in range(n_datapoints):
            model.step(np.random.standard_normal((1, 1)))

        nfoursid = NFourSID(
            model.to_dataframe(),
            model.y_column_names,
            input_columns=model.u_column_names,
            num_block_rows=2
        )
        nfoursid.subspace_identification()
        identified_model, covariance_matrix = nfoursid.system_identification(rank=1)

        # matrices `a` and `d` don't have freedom of choice: they should be fitted well
        self.assertTrue(is_slightly_close(.5, identified_model.a))
        self.assertTrue(is_slightly_close(.8, identified_model.d))
        self.assertTrue(np.all(is_slightly_close(0, covariance_matrix)))

    def test_type_errors(self):
        dt = 0.01
        # time from 0 to 0.99 seconds
        time = np.arange(100)*dt
        # step input
        input_data = np.ones(100)
        # simple first-order system with time constant 0.1
        output_data = np.ones(100) - np.exp(-np.arange(100) * dt / 0.1)
        data = pd.DataFrame({"u": input_data, "y": output_data, "t": time})

        self.assertRaises(TypeError, NFourSID, data, output_columns="y", input_columns=["u"])
        self.assertRaises(TypeError, NFourSID, data, output_columns=["y"], input_columns="u")
        self.assertRaises(TypeError, NFourSID, input_data, output_columns=["y"], input_columns=["u"])

        nfoursid = NFourSID(data, output_columns=["y"], input_columns=["u"])
        nfoursid.subspace_identification()
        identified_model, covariance_matrix = nfoursid.system_identification(rank=1)
        self.assertTrue(is_slightly_close(np.exp(-dt/0.1), identified_model.a))
        self.assertTrue(np.all(is_slightly_close(0, covariance_matrix)))

def is_slightly_close(matrix, number):
    return np.isclose(matrix, number, rtol=0, atol=1e-3)
