"""
Module for running all tests in the test directory
"""
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from workspace.src.tests.TestsCFFUN import *
from workspace.src.tests.TestsRBM import *


if __name__ == '__main__':
    unittest.main()

