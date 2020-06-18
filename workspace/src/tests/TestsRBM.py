import unittest
from unittest import expectedFailure
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.insert(0, "/home/yakkus/github/jack-fork/hclearn/workspace/src/main")

# framework selector, to be able to swap implementations easily.
rbm = None
framework = "tensorflow"
if framework is "tensorflow":
    print("[RBM]Using Tensorflow Version")
    import workspace.src.main.tensor.TensorflowRBM as rbm
    rbm.tf.debugging.set_log_device_placement(True)
else:
    print("[RBM]Using Numpy Version")
    import workspace.src.main.rbm as rbm


class BoltzmannProbsTests(unittest.TestCase):
    """
    Tests for the function rbm#boltzmannProbs
    Test List
        - 2x2 weight matrix and input vector -done
        - 1000x1000 weight matrix and input vector - done
        - Does not error with imaginary numbers 2x2 - done
        - Does not error with imaginary numbers 1000x1000 - done
        - sanity check: Max value is <= 1 - done
        - sanity check: Min value is >= 0 - done
        - Error if input vector is None - done
        - Error if weight matrix is None - done
        - Error if inputs are empty - done
        - Error if inputs are NaN - done
        - Error if inputs are Inf - done
        - Error if result is NaN - done
        - If inputs are too large, overflow occurs, which should error - done
        - If X input is scalar, result should evaluate to softmax?
    """

    def setUp(self) -> None:
        """
        Defines the initial Weight and input matrices
        :return:
        """
        np.random.seed(1000)
        self.initial_weights_2 = np.array([[.1, .2], [.3, .4]])
        self.initial_inputs_2 = np.array([.5, .6])
        self.initial_weights_1000 = np.random.uniform(-1, 1, (1000, 1000))
        self.initial_inputs_1000 = np.random.uniform(-1, 1, 1000)

    def test_two_by_two(self):
        """
        Checks the functionality on a 2x2 weight matrix and 1x2 input vector
        Should pass
        :return:
        """
        calculated_results = rbm.boltzmannProbs(self.initial_weights_2, self.initial_inputs_2)
        actual_results = np.array([0.54239794, 0.5962827])
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[BoltzmannProbs] Calculated results are not equal to actual results to 7 decimal places.")

    def test_thousand_by_thousand(self):
        """
        Checks the functionality on a 1000x1000 weight matrix and 1x1000 input vector
        Should pass
        :return:
        """
        calculated_results = rbm.boltzmannProbs(self.initial_weights_1000, self.initial_inputs_1000)
        numer = np.exp(np.dot(self.initial_weights_1000, self.initial_inputs_1000))
        actual_results = numer/(numer+1)
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[BoltzmannProbs] Calculated results are not equal to actual results to 7 decimal places.")

    def test_complex_two_by_two(self):
        """
        If the input is complex, i.e. has an imaginary component, check that we return a complex number that has an
        absolute minimum value of 0 and absolute maximum value of 1
        :return:
        """
        input_weight_complex = self.initial_weights_2 + self.initial_weights_2*np.random.randn(1)*1j
        input_input_complex = self.initial_inputs_2 + self.initial_inputs_2*np.random.randn(1)*1j
        result = rbm.boltzmannProbs(input_weight_complex, input_input_complex)

        self.assertTrue(np.iscomplexobj(result), msg="[BoltzmannProbs] Result of complex input does not return a complex number.")
        self.assertLessEqual(np.max(np.absolute(result)), 1, msg="[BoltzmannProbs] Absolute value of complex input is greater than 1.")
        self.assertGreaterEqual(np.min(np.absolute(result)), 0, msg="[BoltzmannProbs] Absolute value of complex input is less than 0.")

    def test_complex_thousand_by_thousand(self):
        """
        If the input is complex, i.e. has an imaginary component, check that we return a complex number that has an
        absolute minimum value of 0 and absolute maximum value of 1
        :return:
        """
        input_weight_complex = self.initial_weights_1000 + self.initial_weights_1000*np.random.uniform(-1e-5, 1e-5)*1j
        input_input_complex = self.initial_inputs_1000 + self.initial_inputs_1000*np.random.uniform(-1e-5, 1e-5)*1j
        result = rbm.boltzmannProbs(input_weight_complex, input_input_complex)
        # TODO should we move these asserts to different test functions? From Osherove, R. it's generally considered
        # TODO good to have one logical concept per test, and this can be considered 3, as indicated by the messages.
        self.assertTrue(np.iscomplexobj(result), msg="[BoltzmannProbs] Result of complex input does not return a complex number.")
        self.assertLessEqual(np.max(np.absolute(result)), 1, msg="[BoltzmannProbs] Absolute value of complex input is greater than 1.")
        self.assertGreaterEqual(np.min(np.absolute(result)), 0, msg="[BoltzmannProbs] Absolute value of complex input is less than 0.")

    def test_max_value(self):
        """
        Assumes that something has gone weird in the training process with respect to the weights and/or inputs, resulting
        in them being abnormally high. As this function should return probabilities, we need to ensure the maximum value of
        the outputs <= 1.
        Should pass
        :return:
        """
        initial_weights = 0.5*np.ones((1000, 1000))
        initial_inputs = 0.5*np.ones((1000, 1))
        calculated_results = rbm.boltzmannProbs(initial_weights, initial_inputs)
        self.assertLessEqual(np.max(calculated_results), 1, "[BoltzmannProbs] Maximum Probability is greater than 1")

    def test_min_value(self):
        """
        Assumes that something has gone weird in the training process with respect to the weights and/or inputs, resulting
        in them being abnormally low. As this function should return probabilities, we need to ensure the minimum value of
        the outputs >= 0.
        Should pass
        :return:
        """
        initial_weights = np.zeros((1000, 1000))
        initial_inputs = np.zeros((1000, 1))
        calculated_results = rbm.boltzmannProbs(initial_weights, initial_inputs)
        self.assertGreaterEqual(np.min(calculated_results), 0, "[BoltzmannProbs] Minimum Probability is less than 0")

    def test_weight_input_empty(self):
        """
        Should return 0.5 if inputs to function are empty
        :return:
        """
        NoneResult = rbm.boltzmannProbs(np.array([[],[]]), np.array([]))
        assert_array_equal(NoneResult, 0.5, err_msg="[BoltzmannProbs] None inputs leads to NaN values")

    def test_inputs_nan(self):
        """
        Should return NaN if inputs to function are NaN
        :return:
        """
        nan_result = rbm.boltzmannProbs(self.initial_weights_2*np.nan, self.initial_inputs_2*np.nan)
        assert_array_equal(nan_result, np.nan, err_msg="[BoltzmannProbs] NaN inputs results in NaN values")

    def test_inputs_inf(self):
        """
        Should return NaN if inputs to function are Inf
        :return:
        """
        inf_result = rbm.boltzmannProbs(self.initial_weights_2*np.inf, self.initial_inputs_2*np.inf)
        assert_array_equal(inf_result, np.nan, err_msg="[BoltzmannProbs] Inf inputs results in NaN values")

    def test_scalar_x(self):
        """
        If input x is scalar, return x*W
        :return:
        """
        scalar_results = rbm.boltzmannProbs(self.initial_weights_2, 1.0)
        assert_almost_equal(scalar_results, np.array([[0.524979187, 0.549833997],[0.574442517, 0.59868766]]),
                           err_msg="[BoltzmannProbs] Scalar input does not provide correct values")

    @expectedFailure
    def test_overflow(self):
        """
        If a memory overflow occurs, meaning that the dot products is greater than 1000, then result should be nan, this
        should probably be reported in main code, through a debug message.
        :return:
        """
        inf_result = rbm.boltzmannProbs(np.abs(self.initial_weights_1000)*20, np.abs(self.initial_inputs_1000))
        assert_almost_equal(inf_result, np.ones(inf_result.shape),
                            err_msg="[BoltzmannProbs]Overflow has occurred, resulting in NaN value. This is expected "
                                    "behaviour, however if this has occurred then you should look at your input values")

    @expectedFailure
    def test_weight_input_none(self):
        """
        If the input to the weights parameter is none, we should expect an error to occur
        :return:
        """
        calculated_result = rbm.boltzmannProbs(None, self.initial_inputs_2)
        self.assert_(calculated_result != np.nan, "[BoltzmannProbs] Input to Weight Matrix is None.")

    @expectedFailure
    def test_x_input_none(self):
        """
        If the input to the weights parameter is none, we should expect an error to occur
        :return:
        """
        calculated_result = rbm.boltzmannProbs(self.initial_weights_2, None)
        self.assert_(calculated_result != np.nan, "[BoltzmannProbs] Input to x vector is None.")


class HardThresholdTests(unittest.TestCase):
    """
    Tests for the function rbm#boltzmannProbs
    Test List
        - 2x2 weight matrix and input vector -done
        - 1000x1000 weight matrix and input vector - done
        - Does not error with imaginary numbers 2x2 - done
        - Does not error with imaginary numbers 1000x1000 - done
        - sanity check: Max value is <= 1 - done
        - sanity check: Min value is >= 0 - done
        - Error if input vector is None - done
        - Error if weight matrix is None - done
        - Error if inputs are empty - done
        - Error if inputs are NaN - done
        - Error if inputs are Inf - done
        - Error if result is NaN - done
        - If inputs are too large, overflow occurs, which should error - done
        - If X input is scalar, result should evaluate to softmax?
    """

    def setUp(self) -> None:
        """
        Defines the initial Weight and input matrices
        :return:
        """
        np.random.seed(1000)
        self.initial_2 = np.array([[.1, .8], [.9, .2]])
        self.initial_1000 = np.random.uniform(0, 1, (1000, 1000))

    def test_two_by_two(self):
        """
        Checks the functionality on a 2x2 weight matrix and 1x2 input vector
        Should pass
        :return:
        """
        calculated_results = rbm.hardThreshold(self.initial_2)
        actual_results = np.array([[0, 1], [1, 0]])
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[HardThreshold] Calculated results are not equal to actual results to 7 decimal places.")

    def test_thousand_by_thousand(self):
        """
        Checks the functionality on a 1000x1000 weight matrix and 1x1000 input vector
        Should pass
        :return:
        """
        calculated_results = rbm.hardThreshold(self.initial_1000)
        actual_results = (self.initial_1000 > 0.5)*1
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[HardThreshold] Calculated results are not equal to actual results to 7 decimal places.")

    def test_complex_two_by_two(self):
        """
        If the input is complex, i.e. has an imaginary component, check that we return a complex number that has an
        absolute minimum value of 0 and absolute maximum value of 1
        :return:
        """
        input_2_complex = self.initial_2 + self.initial_2*np.random.randn(1)*1j
        result = rbm.hardThreshold(input_2_complex)

        self.assertFalse(np.iscomplexobj(result), msg="[HardThreshold] Result of complex input returns a complex number.")
        self.assertLessEqual(np.max(np.absolute(result)), 1, msg="[HardThreshold] Absolute value of complex input is greater than 1.")
        self.assertGreaterEqual(np.min(np.absolute(result)), 0, msg="[HardThreshold] Absolute value of complex input is less than 0.")

    def test_complex_thousand_by_thousand(self):
        """
        If the input is complex, i.e. has an imaginary component, check that we return a complex number that has an
        absolute minimum value of 0 and absolute maximum value of 1
        :return:
        """
        input_1000_complex = self.initial_1000 + self.initial_1000*np.random.uniform(-1e-5, 1e-5)*1j
        result = rbm.hardThreshold(input_1000_complex)
        # TODO should we move these asserts to different test functions? From Osherove, R. it's generally considered
        # TODO good to have one logical concept per test, and this can be considered 3, as indicated by the messages.
        self.assertFalse(np.iscomplexobj(result), msg="[HardThreshold] Result of complex input returns a complex number.")
        self.assertLessEqual(np.max(np.absolute(result)), 1, msg="[HardThreshold] Absolute value of complex input is greater than 1.")
        self.assertGreaterEqual(np.min(np.absolute(result)), 0, msg="[HardThreshold] Absolute value of complex input is less than 0.")


class ArgMaxsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_2 = np.array([[0.1, 0.2],[0.5, 0.4]])
        self.matrix_1000 = np.random.uniform(0, 1, (1000,1000))

    def test_two_matrix(self):
        """
        Checks the functionality on a 2x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.argmaxs(self.matrix_2)
        actual_results = np.zeros(self.matrix_2.shape)
        idxs = np.argmax(self.matrix_2, axis=1)
        rows = np.arange(0, len(idxs))
        actual_results[rows, idxs]=1
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is rbm.tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")

    def test_thousand_matrix(self):
        """
        Checks the functionality on a 1000x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.argmaxs(self.matrix_1000)
        actual_results = np.zeros(self.matrix_1000.shape)
        idxs = np.argmax(self.matrix_1000, axis=1)
        rows = np.arange(0, len(idxs))
        actual_results[rows, idxs]=1
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is rbm.tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")


class AddBiasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_2 = np.array([[0.1, 0.2],[0.5, 0.4]])
        self.matrix_1000 = np.random.uniform(0, 1, (1000,1000))

    def test_two_matrix(self):
        """
        Checks the functionality on a 2x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.addBias(self.matrix_2)
        actual_results = np.hstack((self.matrix_2, np.ones((self.matrix_2.shape[0],1))))
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")

    def test_thousand_matrix(self):
        """
        Checks the functionality on a 2x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.addBias(self.matrix_1000)
        actual_results = np.hstack((self.matrix_1000, np.ones((self.matrix_1000.shape[0],1))))
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")


class StripBiasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_2 = np.array([[0.1, 0.2],[0.5, 0.4],[1.,1.]])
        self.matrix_1000 = np.hstack((np.random.uniform(0, 1, (1000,1000)), np.ones((1000,1))))

    def test_two_matrix(self):
        """
        Checks the functionality on a 2x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.stripBias(self.matrix_2)
        actual_results = self.matrix_2[:, 0:-1]
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")

    def test_thousand_matrix(self):
        """
        Checks the functionality on a 2x2 vector
        Should pass
        :return:
        """
        calculated_results = rbm.stripBias(self.matrix_1000)
        actual_results = self.matrix_1000[:, 0:-1]
        # Move to numpy array, although tensorflow might do this automagically.
        if type(calculated_results) is tf.Tensor:
            calculated_results = calculated_results.numpy()
        assert_almost_equal(calculated_results, actual_results,
                            err_msg="[ArgMaxs] Calculated results are not equal to actual results to 7 decimal places.")


class TrainPriorBiasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hids_rand_vector = np.random.uniform(0, 1, (1000,1))
        self.hids_deter_vector = np.array([[0,0,0.1,0.2,0.3,0.4,0.6,0.7,1,0.9]])
        self.hids_deter_matrix = np.array([[0,0,0.1,0.2,0.3],[0.4,0.6,0.7,1,0.9]])
        self.hids_rand_matrix = np.random.uniform(0, 1, (1000,1000))

    def test_retain_shape(self):
        calculated_resultV = rbm.trainPriorBias(self.hids_deter_vector)
        if type(calculated_resultV) is tf.Tensor:
            calculated_resultV = calculated_resultV.numpy()
        assert_array_equal(calculated_resultV.shape, (11))
        calculated_resultM = rbm.trainPriorBias(self.hids_rand_matrix)
        if type(calculated_resultM) is tf.Tensor:
            calculated_resultM = calculated_resultM.numpy()
        assert_array_equal(calculated_resultM.shape, (1001))

    def test_no_inf(self):
        calculated_resultV = rbm.trainPriorBias(self.hids_deter_vector)
        if type(calculated_resultV) is not np.ndarray:
            calculated_resultV = calculated_resultV.numpy()
        self.assertFalse(np.isinf(calculated_resultV.any()))
        calculated_resultM = rbm.trainPriorBias(self.hids_rand_matrix)
        if type(calculated_resultM) is not np.ndarray:
            calculated_resultM = calculated_resultM.numpy()
        self.assertFalse(np.isinf(calculated_resultM.any()))

    def test_no_nan(self):
        calculated_resultV = rbm.trainPriorBias(self.hids_deter_vector)
        if type(calculated_resultV) is not np.ndarray:
            calculated_resultV = calculated_resultV.numpy()
        self.assertFalse(np.isnan(calculated_resultV.any()))
        calculated_resultM = rbm.trainPriorBias(self.hids_rand_matrix)
        if type(calculated_resultM) is not np.ndarray:
            calculated_resultM = calculated_resultM.numpy()
        self.assertFalse(np.isnan(calculated_resultM.any()))


class TrainWTests(unittest.TestCase):
    """

    """


test_cases = [
    BoltzmannProbsTests(),
    HardThresholdTests(),
    ArgMaxsTests(),
    AddBiasTests(),
    StripBiasTests(),
    TrainPriorBiasTests(),
    TrainWTests()
    ]

rbm_suite = unittest.TestSuite(test_cases)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    res = runner.run(test=rbm_suite)
    print(res)