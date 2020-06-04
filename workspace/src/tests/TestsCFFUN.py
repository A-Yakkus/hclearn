import unittest
from unittest import expectedFailure
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal


cffun = None
framework = "tensorflow"
if framework is "tensorflow":
    print("Using Tensorflow Version")
    import workspace.src.main.tensor.Tensorflowcffun as cffun
    print("warmup:",(cffun.tf.constant(5)))
else:
    print("Using Numpy Version")
import workspace.src.main.cffun as cffun

class FuseTests(unittest.TestCase):
    """

    """

class LagTests(unittest.TestCase):

    def setUp(self) -> None:
        self.initial_one_by_two = np.random.uniform(0, 1, (2, 1))
        self.initial_one_by_thousand = np.random.uniform(0, 1, (1000, 1))
        self.initial_two_by_two = np.random.uniform(0, 1, (2, 2))
        self.initial_thousand_by_thousand = np.random.uniform(0, 1, (1000, 1000))

    def test_one_by_two_once(self):
        results = cffun.lag(self.initial_one_by_two, 1)
        assert_array_equal(results, self.initial_one_by_two, "[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

    def test_one_by_thousand_once(self):
        results = cffun.lag(self.initial_one_by_thousand, 1)
        assert_array_equal(results, self.initial_one_by_thousand, "[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

    def test_two_by_two_once(self):
        results = cffun.lag(self.initial_two_by_two, 1)
        actual = np.vstack((np.repeat([self.initial_two_by_two[0, :]], 1, 0), self.initial_two_by_two))[:-1, :]
        assert_array_equal(results, actual, "[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

    def test_thousand_by_thousand_once(self):
        results = cffun.lag(self.initial_thousand_by_thousand, 1)
        actual = np.vstack((np.repeat([self.initial_thousand_by_thousand[0, :]], 1, 0), self.initial_thousand_by_thousand))[:-1, :]
        assert_array_equal(results, actual, "[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

    def test_thousand_by_thousand_hundred(self):
        results = cffun.lag(self.initial_thousand_by_thousand, 100)
        actual = np.vstack((np.repeat([self.initial_thousand_by_thousand[0, :]], 100, 0), self.initial_thousand_by_thousand))[:-100, :]
        assert_array_equal(results, actual, "[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")