import unittest
import matplotlib.pyplot as plt
from DGStateAlan import *


class TestEncoding(unittest.TestCase):
    def setUp(self):
        # Use a seed so results are consistent
        SEED = 2942875  # 95731  # 73765
        random.seed(SEED)    # careful, these are different RNGs!
        np.random.seed(SEED)

        # Make fake data and noisy copy
        self.fakeSURF = np.random.randint(0, 2, (10,))
        noiseProb = 0.1

        self.noisyFakeSURF = addNoise(self.fakeSURF, noiseProb)

        self.N = 4
        self.X = 3
        # Choose the noisy data (so we're not relying on the data seed to make this work!)
        # self.chosenNoisyData = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 1])
        self.chosenNoisyData = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.semantics = np.array([[2, 1, 7], [0, 5, 2], [5, 2, 8], [3, 5, 0]])

        # Test ECDGweights
        # Weights are as so where sf = surf feature directly from the input of ec, and ou = output unit which is the
        # neuron that will fire in response to winner takes all
        #       /                   /
        #      /___________________/ b2
        #     /                   /
        #    /___________________/ b1
        #    |sf0____sf1_____sf2_|
        # ou0| x      x       x  |
        #    |                   |
        # ou1| x      x       x  |
        #    |                   | |/
        # ou2| x      x       x  | |
        #    |___________________|/

        self.encodedTestWeights = np.zeros(shape=(self.N, self.X, self.X))
        self.encodedTestWeights[2, 0, 1] = 0.5
        self.encodedTestWeights[2, 0, 2] = 0.3
        self.encodedTestWeights[2, 2, 2] = 0.75
        self.encodedTestWeights[0, 0, 0] = 0.5
        self.encodedTestWeights[0, 1, 0] = 0.75
        self.encodedTestWeights[0, 2, 2] = 0.25

        self.CA3CA1TestWeights = np.zeros(shape=(self.N, self.X, self.X))

        # Make dentate gyrus
        self.dgh = DGHelper(numOfSurfFeatures=len(self.chosenNoisyData), initialECDGweights=self.encodedTestWeights.copy(), initialCA3CA1weights=self.CA3CA1TestWeights.copy(),  X=self.X, N=self.N, initialSemantics=self.semantics)
        self.dg = DGState(self.chosenNoisyData, None, self.dgh, unittesting=1)

    def test_semantic_values(self):
        data = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        semantics = np.array([[2, 1, 7], [0, 5, 2], [5, 2, 8], [3, 5, 0]])
        svdgh = DGHelper(initialSemantics=semantics)
        sv = svdgh.getSemanticValues(data)
        realSemanticValues = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
        self.assertTrue(np.all(sv == realSemanticValues))

    def test_making_DGState(self):
        self.assertIsNotNone(self.dg)
        self.assertIsNotNone(self.dgh)

    def test_smartCollapseMatrix(self):
        data = np.array([[1, 1, 2, 1], [4, 6, 7, 1], [5, 2, 1, 1]])
        resultdata = smartCollapseMatrix(data)
        correctResultdata = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        self.assertTrue(np.all(resultdata == correctResultdata), "Winner takes all works, two matrices are equivalent")

    # @unittest.skip("Saving time whilst testing other")
    def test_dot_product(self):
        W = np.array([[200, 500, 0], [0, 0, 100], [600, 0, 0]])
        A = np.array([0, 1, 1])
        dotted = np.dot(W, A)
        collapsed = smartCollapse(dotted)
        """
        print("W:\n%s" % W)
        print("A:\n%s" % A)
        print("W dot A:\n%s" % dotted)
        print("W multiplied A:\n%s" % (W*A))
        print("W multiplied A and collapsed:\n%s" % smartCollapse(W*A))
        print("Smart collapse:\n%s" % collapsed) 
        print("Multiplied:\n%s" % (W*collapsed))
        print("Tile test:\n%s" % np.transpose(np.tile((np.array([0,1,0])),(3,1))))
        """
        self.assertTrue(np.all((W*A) == np.array([[0, 500, 0],[0, 0, 100],[0, 0, 0]])))

    def test_encode_type(self):
        encodedData = self.dg.toVectorSurfOnly()
        self.assertTrue(encodedData.ndim == 1)
        self.assertTrue(len(encodedData) == self.X*self.N)
        self.assertTrue(encodedData.dtype == np.int8)
        self.assertTrue(np.sum(encodedData) == self.N,
                        "Encoded data should be sparse and only have one winner per block")

    def test_encode_ability(self):
        # Note we are still using the np.random.seed() for this to work as this is where out input navigation is coming
        # from!
        # print("noisy data to be encoded:\n%s" % self.chosenNoisyData)
        # print("Semantics Indices:\n%s" % self.dg.semanticIndices)
        # print("Semantic values:\n%s" % self.dg.semanticValues)
        encodedData = self.dg.toVectorSurfOnly()
        # print("Encoded data:\n%s" % encodedData)
        correctResultData = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        self.assertTrue(np.all(encodedData == correctResultData), "In this case\n \
                the weight between surf feature 2 and output unit 0 block 0  has a positive weight of 0.5, activation \
            should be 0.5 as feature 2 is active\n and the weight between surf feature 2 and output unit 1 block 0  \
            has a positive weight of 0.75, activation should be 0.75 which wins over first one\n and the weight \
            between surf feature 2 and output unit 0 block 2, has a positive weight of 0.5, activation should be 0.5 \
            as feature 2 is active\n and the weight between surf feature 8 and output unit 0 block 2, has a positive \
            weight of 0.3, activation should be 0.3 as feature 8 is active\n and the weight between surf feature 8 and \
            output unit 2 block 2, has a positive weight of 0.75, activation should be 0.75 as feature 8, however \
            because the previous two ECDGweights both go into output unit 0, its overall activation is 0.8 which beats \
            0.75 thus output unit 0 wins")

    def test_learning(self):
        encodingdgh = DGHelper(numOfSurfFeatures=len(self.chosenNoisyData), initialECDGweights=self.encodedTestWeights.copy(), initialCA3CA1weights=self.CA3CA1TestWeights.copy(), X=3, N=4)
        encodingdgh.learn(self.chosenNoisyData)
        self.assertGreater(encodingdgh.ECDGweights[2, 0, 1], self.encodedTestWeights[2, 0, 1])
        self.assertGreater(encodingdgh.ECDGweights[2, 0, 2], self.encodedTestWeights[2, 0, 2])
        self.assertEqual(encodingdgh.ECDGweights[2, 2, 2], self.encodedTestWeights[2, 2, 2],
                         "A bit fucked up because we are now normalising...")
        self.assertEqual(encodingdgh.ECDGweights[3, 0, 1], self.encodedTestWeights[3, 0, 1])
        # Since all ACTIVE input units ECDGweights connecting to the winning output are increased, this is also
        # increased as it contributed to the units activation
        self.assertGreater(encodingdgh.ECDGweights[0, 1, 0], self.encodedTestWeights[0, 1, 0])
        self.assertEqual(encodingdgh.ECDGweights[0, 0, 0], self.encodedTestWeights[0, 0, 0])

    @unittest.skip("Saving time whilst testing other")
    def test_multiple_learning(self):
        trials = 300
        average = 20  # 20
        X = 3  # 4
        N = 15  # 25

        accuracyOfModel = 0
        for x in range(average):
            # Chosen as a guess would give an accuracy of 50%
            initialData = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            np.random.shuffle(initialData)
            # print("initial data:\n%s" % initialData)
            # initialECDGWeights = np.random.rand(N, X, X)/10
            # initialECDGWeights = np.zeros(shape=(N, X, X))
            # CA3CA1TestWeights = np.zeros(shape=(N, X, X))

            # def __init_(self, initialECDGWeights=False, initialCA3CA1weights=False, X=3, N=4, initialSemantics=False):
            encodingdgh = DGHelper(numOfSurfFeatures=len(initialData), X=X, N=N)

            # print("Initial ECDGweights:\n%s" % (np.around(initialECDGWeights, 3)))

            # print("Semantics:\n%s" % encodingdg.semanticIndices)
            noiseProb = 0.1

            # Log activations used to look at later
            activationsUsed = np.zeros((trials, initialData.shape[0]), np.int8)

            # Generate data to learn with
            for trial in range(trials):
                newData = addNoise(initialData, noiseProb)
                activationsUsed[trial] = newData

            # Preferable to learn in two separate phases as otherwise CA3CA1 will learn noisy mappings and slowly get
            # better as ECDG connections get better
            # Train ECDGweights
            for trial in range(trials):
                encodingdgh.learn(newData, False)
            # print("Final ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))

            # Train CA3CA1weights
            for trial in range(trials):
                encodingdgh.learn(newData, True)
                # ECDGweights = smartCollapseMatrix(encodingdg.ECDGweights)
                # print("Learnt ECDGweights after %d trials:\n%s" % (trials, encodingdg.ECDGweights))

            probabilitiesOfFiring = np.ones((1, X*N))*0.5
            encodingdg = DGState(initialData, None, encodingdgh, unittesting=1)
            decoded = encodingdg.decode(probabilitiesOfFiring)
            # print("After ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
            # print("Decoded:\n%s" % decoded)
            # print("Orignial:\n%s" % initialData)
            # print("All activations used:\n%s" % activationsUsed)
            accuracyOfModel += accuracy(decoded, initialData)

        # print("ECDG Weights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
        # print("CA3CA1 Weights:\n%s" % (np.around(encodingdg.CA3CA1weights, 3)))
        # print("Semantics:\n%s" % encodingdg.semanticIndices)
        # print("initialData:\n%s" % initialData)
        accuracyOfModel /= average
        accuracyOfModel = accuracyOfModel*100
        # print("Accuracy: %f%%" % accuracyOfModel)
        self.assertGreater(accuracyOfModel, 0.5, "Any less than 50% accuracy means it is worse than just guessing")

        # Test my equivalence of ECDG and CA3CA1 weights theory
        # self.assertTrue(np.allclose(np.around(encodingdg.ECDGweights, 3), np.around(encodingdg.CA3CA1weights, 3)))

    def test_decode(self):
        # This is how they will be originally encoded, since the probabilities of firing from the boltzmann are
        # provided this is ignored (except for the semantics)
        #                                  2       1       7       0    5     2   5    2    8      3    5       0
        # partiallyEncoded = np.array([0.5*1, 1*0.75, 0*0.25, 1*0.25, 0*0, 1*0, 0*0, 1*0, 1*0, 0*0.5, 0*0, 1*0.75])
        # fullyEncoded = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])

        # Here are a set of possible probabilities of firings (given by a boltzmann machine)
        # These are the outputs that are winning! thats why we transpose the matrix!
        probabilitiesOfFiring = np.array([0.9, 0.2, 0.1, 0.6, 0.5, 0.1, 0.6, 0.5, 0.5, 0.05, 0.1, 0.25])
        # I.e. given that output unit 0 of block 2 is on, what are the probabilities that feature 2 and 8 are active,
        # if they are more than 50%, they are on
        # Here are the active neurons chosen after WTA is applied
        collapsed = np.array([[1, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        #        dg.CA1CA3weights = np.array([

        # When the output neurons activities are combined with the transpose of the ECDGweights (bringing it back to
        # probability that each is firing?)
        outputprobabilityofunitsfiring = np.array([[0.5,   0.,    0.],
                                                   [0.,    0.,    0.],
                                                   [0.,    0.5,   0.3],
                                                   [0.,    0.,    0.]])
        # Here is how these values were collected, and their corresponding surf feature indices used for decoding (the
        # semantics of the encoding)
        #                                             2     1    7    0    5    2    5     2      8     3    5    0
        whereoutputsprobabilitycamefrom = np.array([1*0.5, 0*0, 0*0, 1*0, 0*0, 1*0, 0*0, 1*0.5, 1*0.3, 0*0, 0*0, 0*0])

        # Does the decoder use and AND? if its the winner of any of the competitions, then it should be on.
        # Here are the decoded values of the cleaned neurons using the probability that they will be on and their
        # semantics to decode
        fullyDecoded = ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        # Decoding should first calculate which neurons won, then convert back to EC space
        decoded = self.dgh.decode(probabilitiesOfFiring)
        # print("Should be:\n%s\nIs:\n%s" % (fullyDecoded,decoded))
        # Fails as we are no longer using ECDG transpose to decode
        self.assertTrue(np.all(decoded == fullyDecoded))

    def test_CA3CA1learning(self):
        # self.chosenNoisyData = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        # self.semantics = np.array([[2,1,7],[0,5,2],[5,2,8],[3,5,0]])

        # Weights to make feature 2 active, so the input and the output are identical, thus no weight changes
        w2on = np.array(
            [[[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    0.,    0.],
              [0.,    0.,    0.]]])

        # For learning CA3CA1 we don't need the weights for ECDG
        ecw = np.zeros(shape=(w2on.shape))

        surfFeaturesValues = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2on.copy(), X=3, N=4, initialSemantics=self.semantics)
        # encodingdg = DGState(self.chosenNoisyData, None, encodingdgh, unittesting=1)
        dgvalues = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 0, 0]])

        # sv = encodingdg.getSemanticValues(surfFeaturesValues, self.semantics)
        # The raw input should be given to CA3CA1, and it will get the semantic values OR it needs changing in that
        # method and the sv can be given to it
        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        # print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertTrue(np.all(w2on == encodingdgh.CA3CA1weights), "If one block's weights applied with the activation \
        of the DG representation thinks a surffeature is present it is regarded as present")

        w2off = np.array(
            [[[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    0.,    1.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    0.,    0.],
              [0.,    0.,    0.]]])

        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2off.copy(), X=3, N=4, initialSemantics=self.semantics)
        # encodingdg = DGState(self.chosenNoisyData, None, ecw, w2off.copy(), X=3, N=4, semantics=self.semantics,
        # unittesting=1)
        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        # print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertFalse(np.all(w2off == encodingdgh.CA3CA1weights), "If none of the block's weights applied with the \
        activation of the DG representation think a surffeature is present it is not regarded as present")

        # So if one block thinks that a neuron is on activation is over 0.5) it is counted as being active, but
        # probabilities in combination will not work
        # I.e if block 1 thinks neuron is on with 0.25 certainty, and block 2 thinks neuron is on with 0.25 certainty,
        # it is not concidered active
        w2halfOn = np.array(
            [[[0.25,  0.,    0.],
              [0.,    0.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    1.,    0.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    0.5,   1.],
              [0.,    0.,    0.]],
             [[0.,    0.,    0.],
              [0.,    0.,    0.],
              [0.,    0.,    0.]]])

        # encodingdg = DGState(self.chosenNoisyData, None, ecw, w2halfOn.copy(), X=3, N=4, semantics=self.semantics,
        # unittesting=1)
        encodingdgh = DGHelper(numOfSurfFeatures=len(surfFeaturesValues), initialECDGweights=ecw, initialCA3CA1weights=w2halfOn.copy(), X=3, N=4, initialSemantics=self.semantics)

        encodingdgh.learnCA3CA1weights(surfFeaturesValues, dgvalues)
        # print("PERCEPTRON ENCODED WEIGHTS:\n%s" % encodingdg.CA3CA1weights)
        self.assertTrue(np.all(w2halfOn == encodingdgh.CA3CA1weights), "If any of the blocks think that a surf feature \
        is present, it is regarded as present, but not if two probabilities combine, i.e. 0.25 probability of 2 being \
        on, and 0.25 probability of it being on from two blocks")

    def test_learningCA3CA1weights(self):
        # Learn one piece of data, then encode it, and decode it
        initialData = np.array([0,0,0,0,0,1,1,1,1,1])
        X=4
        N=25
        dgh = DGHelper(numOfSurfFeatures=len(initialData), X=X, N=N)

        # learningDG = DGState(initialData, None, dgh, unittesting=1)
        # print("Initial ECDGweights:\n%s" % (np.around(initialECDGWeights, 3)))

        # Number of learning cycles to learn the weights
        trials = 100

        # Preferable to learn in two separate phases as otherwise CA3CA1 will learn noisy mappings and slowly get better
        # as ECDG connections get better
        # Train ECDGweights
        for trial in range(trials):
            dgh.learn(initialData, False)

        # Train CA3CA1weights
        for trial in range(trials):
            dgh.learn(initialData, True)

        probabilitiesOfFiring = np.ones((1,X*N))*0.5
        # encoding dg will encode the data, then this data will be decoded with the probabilities of each dg firing
        # given
        decoded = dgh.decode(probabilitiesOfFiring)
        """
        print("Decoded:\n%s" % decoded)
        print("Orignial:\n%s" % initialData)
        print("Final ECDGweights:\n%s" % (np.around(encodingdg.ECDGweights, 3)))
        print("Final CA3CA1weights:\n%s" % (np.around(encodingdg.CA3CA1weights, 3)))
        """
        # After 100 trials the encode -- decode should be learnt
        self.assertTrue(np.all(decoded == initialData))

    def test_normalisation(self):
        w = np.array([[[0,      1,       1],
                       [1,      0,       0.25],
                       [0,      1,       0]],
                      [[0,      1,       1],
                       [0.333,  0.333,   0.333],
                       [0.25,      0.25,    0]]])

        correctRowNormalised = np.array(
            [[[0.,   0.5,    0.5],
              [0.8,    0.,   0.2],
              [0.,     1.,   0.]],
             [[0.,   0.5,    0.5],
              [0.33333333,  0.33333333,  0.33333333],
              [0.5,  0.5,    0.]]])

        rowNormalised = normalise(w, axis=2)

        self.assertTrue(np.allclose(rowNormalised, correctRowNormalised), "Testing normalising rows")

        correctColNormalised = np.array(
            [[[0.,  0.5,  0.8],
              [1.,   0.,  0.2],
              [0.,  0.5,  0.]],
             [[0.,  0.63171194,  0.75018755],
              [0.57118353,  0.21036008, 0.24981245],
              [0.42881647, 0.15792798,  0.]]])
        colNormalised = normalise(w, axis=1)
        self.assertTrue(np.allclose(colNormalised, correctColNormalised), "Testing normalising cols")

        # Must be given either rows or columns, not pages
        self.assertRaises(NameError, normalise, w, 0)

    def test_performanceMeasure(self):
        # Generate initial data
        initialData = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
        noiseLevels = [0.1, 0.2, 0.5]
        figs = [plt.figure(), plt.figure()]
        for noiselevel in noiseLevels:
            numOfImages = 40
            probabilityOfNoise = noiselevel
            presentationOfData = 40
            learningrate = 0.01
            dataBeforeNoise = np.zeros((numOfImages, initialData.shape[0]), dtype=np.int8)
            for image in range(numOfImages):
                np.random.shuffle(initialData)
                dataBeforeNoise[image] = initialData

            # Add noise to the data and save it as new data
            dataAfterNoise = np.zeros(dataBeforeNoise.shape, dtype=np.int8)
            for imageNum, image in enumerate(dataBeforeNoise):
                # np.random.shuffle(initialData)
                # dataAfterNoise[imageNum] = initialData
                dataAfterNoise[imageNum] = addNoise(image,probabilityOfNoise)

            # print (dataBeforeNoise)
            # print (dataAfterNoise)

            # Plot accuracy as graph
            Xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            Ns = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
            A = np.zeros((len(Ns), len(Xs)))
            A = A + Xs
            B = np.zeros((len(Xs), len(Ns)))
            B = np.transpose((B + Ns))
            TA = np.zeros((A.shape[0], B.shape[1]))
            TC = np.zeros((A.shape[0], B.shape[1]))
            for xind, X in enumerate(Xs):
                for nind, N in enumerate(Ns):
                    # FIX: Bug somewhere? why doesnt accuracy decrease when data is effectively completely random, i.e.
                    # theres no correlation between initial and noisy?
                    (totalAccuracy, totalChange)= calculate_performance(dataBeforeNoise, dataAfterNoise, X, N, presentationOfData, learningrate)
                    TA[nind,xind] = totalAccuracy
                    TC[nind,xind] = 100-totalChange
                    # print("Total accuracy with X=%d, N=%d, and the data being learnt over %d presentations: %f" %
                    #   (X,N,presentationOfData,totalAccuracy))
                    # print("Total change between noisy and decoded with X=%d, N=%d, and the data being learnt over %d\
                    #   presentations: %f" % (X,N,presentationOfData,totalChange))

            datatypes = [TA,TC]
            for fignum, fig in enumerate(figs):
                dataType = datatypes[fignum]
                #fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_color_cycle(['red', 'green', 'blue', 'yellow'])
                ax.plot_surface(A, B, dataType, rstride=1, cstride=1, alpha=0.8, cmap=plt.cm.jet,#color=(noiselevel*100,noiselevel*100,noiselevel*100), #
                                linewidth=1, antialiased=True)

                cset = ax.contour(A, B, dataType, zdir='z', offset= 0)
                cset = ax.contour(A, B, dataType, zdir='x', offset= 12)
                cset = ax.contour(A, B, dataType, zdir='y', offset= 90)

                ax.set_xlabel('X')
                ax.set_xlim3d(0, 12)
                ax.set_ylabel('N')
                ax.set_ylim3d(0, 90)
                ax.set_zlabel('Accuracy %')
                ax.set_zlim3d(0, 100)

                # Think of a nice way to plot this 3 graph for several noise levels
                if np.all(dataType == TA):
                    title = "Accuracy in de-noising noisy input when trained on clean input\nData is presented %d \
                    times with a learning rate of %f" % (presentationOfData, learningrate)
                elif np.all(dataType == TC):
                    title = "Accuracy in reconstructing noisy input when trained on clean input\nData is presented %d\
                     times with a learning rate of %f" % (presentationOfData, learningrate)
                fig.suptitle(title, fontsize=12)
        plt.show()


if __name__ == '__main__':
    unittesting = 1
    unittest.main()

