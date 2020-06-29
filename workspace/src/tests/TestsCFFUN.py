import unittest
from unittest import expectedFailure
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import tensorflow as tf

cffun = None
framework = "tensorflow"
if framework is "tensorflow":
    print("[CFFUN]Using Tensorflow Version")
    import workspace.src.main.tensor.Tensorflowcffun as cffun
else:
    print("[CFFUN]Using Numpy Version")
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
        self.initial_nine_two = np.random.uniform(0, 1, (9, 2))

    def test_one_by_two_once(self):
        results = cffun.lag(self.initial_one_by_two, 1)
        actual = np.insert(self.initial_one_by_two, 0, self.initial_one_by_two[0], axis=0)[:-1]
        self.assertEqual(results[0], results[1], msg="[Lag] When Lagging by one, the 0th and 1st elements must match")
        assert_array_equal(results, actual, err_msg="[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

    def test_one_by_thousand_once(self):
        results = cffun.lag(self.initial_one_by_thousand, 1)
        actual = self.initial_one_by_thousand
        actual = np.insert(actual, 0, self.initial_one_by_thousand[0], axis=0)[:-1]
        self.assertEqual(results[0], results[1], msg="[Lag] When Lagging by one, the 0th and 1st elements must match")
        assert_array_equal(results, actual, err_msg="[Lag] When lagging by one on a 2D array with second dim=1, result should be the same")

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


class InvSigTests(unittest.TestCase):
    """

    """
    def setUp(self) -> None:
        self.vector_10 = np.linspace(1e-10, 0.9, 10)
        self.vector_1000 = np.random.uniform(0, 0.9, 1000)
        self.negative_vector = np.random.uniform(-1, 0, 1000)
        self.matrix_2 = np.array([[1e-7,.5],[.9,1e-7]])
        self.matrix_1000 = np.random.uniform(0, 0.9, (1000,1000))

    def test_vector_10(self):
        calculated_results = cffun.invsig(self.vector_10)
        actual_results = -np.log((1./self.vector_10)-1)
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results)

    def test_vector_1000(self):
        calculated_results = cffun.invsig(self.vector_1000)
        actual_results = -np.log((1./self.vector_1000)-1)
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results)

    def test_negative_vector(self):
        calculated_results = cffun.invsig(self.negative_vector)
        actual_results = -np.log((1./self.negative_vector)-1)
        print(calculated_results)
        print(actual_results)
        if hasattr(calculated_results, "numpy"):
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results)
        self.assertFalse(np.isnan(calculated_results.all()))

    def test_matrix_2(self):
        calculated_results = cffun.invsig(self.matrix_2)
        actual_results = -np.log((1./self.matrix_2)-1)
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results)

    def test_matrix_1000(self):
        calculated_results = cffun.invsig(self.matrix_1000)
        actual_results = -np.log((1./self.matrix_1000)-1)
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results)

#@unittest.skipIf(framework is not "tensorflow", "cffun#outer is a replacement for numpy outer. Numpy testing is not necessary")
class OuterTests(unittest.TestCase):
    def setUp(self):
        self.vector_10 = np.linspace(1e-10, 0.9, 10)
        self.vector_1000 = np.random.uniform(0, 0.9, 1000)
        self.negative_vector = np.random.uniform(-1, 0, 1000)
        self.matrix_2 = np.array([[1e-7,.5],[.9,1e-7]])
        self.matrix_100 = np.random.uniform(0, 0.9, (100, 100))
        self.scalar = tf.convert_to_tensor([2.], dtype=tf.double)

    def test_scalar_vector10(self):
        calculatedResults = cffun.outer(self.scalar, self.vector_10)
        actualResults = np.outer(self.scalar, self.vector_10)
        assert_almost_equal(calculatedResults, actualResults)

    def test_scalar_vector1000(self):
        calculatedResults = cffun.outer(self.scalar, self.vector_1000)
        actualResults = np.outer(self.scalar, self.vector_1000)
        assert_almost_equal(calculatedResults, actualResults)

    def test_vector1000_vector10(self):
        calculatedResults = cffun.outer(self.vector_1000, self.vector_10)
        actualResults = np.outer(self.vector_1000, self.vector_10)
        assert_almost_equal(calculatedResults, actualResults)

    def test_negativeVector_vector10(self):
        calculatedResults = cffun.outer(self.negative_vector, self.vector_10)
        actualResults = np.outer(self.negative_vector, self.vector_10)
        assert_almost_equal(calculatedResults, actualResults)

    def test_negativeVector_vector1000(self):
        calculatedResults = cffun.outer(self.negative_vector, self.vector_1000)
        actualResults = np.outer(self.negative_vector, self.vector_1000)
        assert_almost_equal(calculatedResults, actualResults)

    def test_scalar_matrix2(self):
        calculatedResults = cffun.outer(self.scalar, self.matrix_2)
        actualResults = np.outer(self.scalar, self.matrix_2)
        assert_almost_equal(calculatedResults, actualResults)

    def test_vector10_matrix2(self):
        calculatedResults = cffun.outer(self.vector_10, self.matrix_2)
        actualResults = np.outer(self.vector_10, self.matrix_2)
        assert_almost_equal(calculatedResults, actualResults)

    def test_negativeVector_matrix2(self):
        calculatedResults = cffun.outer(self.negative_vector, self.matrix_2)
        actualResults = np.outer(self.negative_vector, self.matrix_2)
        assert_almost_equal(calculatedResults, actualResults)

    def test_matrix100_matrix2(self):
        calculatedResults = cffun.outer(self.matrix_100, self.matrix_2)
        actualResults = np.outer(self.matrix_100, self.matrix_2)
        assert_almost_equal(calculatedResults, actualResults)

    def test_matrix100_matrix100(self):
        calculatedResults = cffun.outer(self.matrix_100, self.matrix_100)
        actualResults = np.outer(self.matrix_100, self.matrix_100)
        assert_almost_equal(calculatedResults, actualResults)


test_cases = [
    #FuseTests(),
    LagTests(),
    InvSigTests(),
    OuterTests()
]

cffun_suite = unittest.TestSuite(test_cases)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    res = runner.run(test=cffun_suite)
    print(res)
