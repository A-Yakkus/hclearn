import random
import os
import sys
import unittest
from unittest import expectedFailure
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from workspace.src.main.makeMaze import makeMaze
from workspace.src.main.location import DictGrids
from workspace.src.main.paths import Paths
import workspace.src.main.learnWeights as lw
from workspace.src.main.DGStateAlan import DGHelper
import tensorflow as tf

class LearnWeightsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Define random Seed to be the same across all rng generators that may be used.
        SEED = 100
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"]=str(SEED)
        os.environ["TF_CUDNN_DETERMINITIC"]='1'
        self.pathing = "../data/"
        self.b_useNewDG = True
        self.learningRate = 0.01
        np.set_printoptions(threshold=sys.maxsize)
        self.N_mazeSize = 3
        self.T = 3000   # trained on 30000   # better to have one long path than mult epochs on overfit little path
        self.b_learnWeights = True
        self.b_plot = False
        self.b_inference = True
        [self.dictSenses, self.dictAvailableActions, self.dictNext] = makeMaze(self.N_mazeSize, self.b_useNewDG)
        self.dictGrids = DictGrids()
        self.path = Paths(self.dictNext, self.N_mazeSize, self.T)
        (self.ecs_gnd, self.dgs_gnd, self.ca3s_gnd) = self.path.getGroundTruthFirings(self.dictSenses, self.dictGrids, self.N_mazeSize)
        lw.pathing = "../data/"
        self.ideal_data = {
            "biases": np.load("../data_100/WB.npy"),
            "odom": np.load("../data_100/WO.npy"),
            "senses": np.load("../data_100/WS.npy"),
            "r": np.load("../data_100/WR.npy")
        }
        self.inference_data = {
            "biases": np.load("../data_100/tWB.npy"),
            "odom": np.load("../data_100/tWO.npy"),
            "senses": np.load("../data_100/tWS.npy"),
            "r": np.load("../data_100/tWR.npy")
        }

    def test_None(self):
        dgHelper = lw.learn(self.path, self.dictSenses, self.dictGrids, self.N_mazeSize, self.ecs_gnd, self.dgs_gnd, self.ca3s_gnd, False, False, False, self.learningRate)
        self.assertIsNone(dgHelper, "[LearnWeights] Learn should be none when all learning types are false.")

    def test_DG(self):
        dgHelper = lw.learn(self.path, self.dictSenses, self.dictGrids, self.N_mazeSize, self.ecs_gnd, self.dgs_gnd, self.ca3s_gnd, False, False, True, self.learningRate)
        self.assertIsInstance(dgHelper, DGHelper, "[LearnWeights] Learn should return an Instance of DGHelper if b_learnDG is true.")

    def test_ideal(self):
        dgHelper = lw.learn(self.path, self.dictSenses, self.dictGrids, self.N_mazeSize, self.ecs_gnd, self.dgs_gnd, self.ca3s_gnd, True, False, False, self.learningRate)
        self.assertIsNone(dgHelper, "[LearnWeights] Learn should be none if b_learnDG is false.")
        biases = np.load("../data/WB.npy")
        senses = np.load("../data/WS.npy")
        odom = np.load("../data/WO.npy")
        r = np.load("../data/WR.npy")
        assert_array_equal(biases.shape, self.ideal_data["biases"].shape)
        assert_array_equal(senses.shape, self.ideal_data["senses"].shape)
        assert_array_equal(odom.shape, self.ideal_data["odom"].shape)
        assert_array_equal(r.shape, self.ideal_data["r"].shape)


    def test_inference(self):
        dgHelper = lw.learn(self.path, self.dictSenses, self.dictGrids, self.N_mazeSize, self.ecs_gnd, self.dgs_gnd, self.ca3s_gnd, False, True, False, self.learningRate)
        self.assertIsNone(dgHelper, "[LearnWeights] Learn should be none if b_learnDG is  false.")
        biases = np.load("../data/tWB.npy")
        senses = np.load("../data/tWS.npy")
        odom = np.load("../data/tWO.npy")
        r = np.load("../data/tWR.npy")
        assert_array_equal(biases.shape, self.inference_data["biases"].shape)
        assert_array_equal(senses.shape, self.inference_data["senses"].shape)
        assert_array_equal(odom.shape, self.inference_data["odom"].shape)
        assert_array_equal(r.shape, self.inference_data["r"].shape)