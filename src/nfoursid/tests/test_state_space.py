import unittest

import numpy as np

from nfoursid.state_space import StateSpace


class TestStateSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.a = np.array([[2]])
        self.b = np.array([[3]])
        self.c = np.array([[5]])
        self.d = np.array([[7]])
        self.k = np.array([[11]])

    def test_output(self):
        state = np.array([[1]])
        u = np.array([[2]])
        e = np.array([[3]])

        model = StateSpace(
            self.a,
            self.b,
            self.c,
            self.d,
            self.k
        )

        result = model.output(state)
        self.assertAlmostEqual(5, result[0, 0])

        result = model.output(state, u=u)
        self.assertAlmostEqual(19, result[0, 0])

        result = model.output(state, e=e)
        self.assertAlmostEqual(8, result[0, 0])

    def test_step(self):
        u = np.array([[2]])
        e = np.array([[3]])

        model = StateSpace(
            self.a,
            self.b,
            self.c,
            self.d,
            self.k
        )

        y = model.step()
        self.assertAlmostEqual(0, y[0, 0])
        y = model.step(u)
        self.assertAlmostEqual(14, y[0, 0])
        y = model.step(u=u)
        self.assertAlmostEqual(44, y[0, 0])
        y = model.step(e=e)
        self.assertAlmostEqual(93, y[0, 0])

        result = model.to_dataframe().to_numpy()
        self.assertTrue(np.all(np.isclose(
            np.array([
                [0, 0],
                [2, 14],
                [2, 44],
                [0, 93]
            ]),
            result
        )))

    def test_autonomous_system(self):
        model = StateSpace(
            self.a,
            np.zeros((1, 0)),
            self.c,
            np.zeros((1, 0)),
            k=np.array([[1]])
        )
        e = np.array([[1]])

        with self.assertRaises(ValueError):
            model.step(np.array([[1]]))
        y = model.step()
        self.assertEqual((1, 1), y.shape)
        self.assertAlmostEqual(0, y[0, 0])
        y = model.step(e=e)
        self.assertAlmostEqual(1, y[0, 0])
        y = model.step(e=e)
        self.assertAlmostEqual(6, y[0, 0])
        y = model.step(e=e)
        self.assertAlmostEqual(16, y[0, 0])
