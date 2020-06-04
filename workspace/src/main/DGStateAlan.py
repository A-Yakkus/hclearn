# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import unittest
import numpy as np
import random
from location import Location


#Use a seed so results are consistent
SEED=2942875  #95731  #73765
random.seed(SEED)    #careful, these are different RNGs!
np.random.seed(SEED)
#unittesting=0

class DGHelper:
    def __init__(self, numOfSurfFeatures = None, initialECDGweights = None, initialCA3CA1weights = None, X = 3, N = 4, initialSemantics = None):
        self.X = X
        self.N = N
        self.numOfEncodedFeatures = self.X*self.N
        self.numOfSurfFeatures = numOfSurfFeatures
        # Train the network on the clean initial data
        if initialECDGweights is None:
            self.ECDGweights = np.random.rand(N, X, X)/10
        else:
            self.ECDGweights = initialECDGweights

        if initialCA3CA1weights is None:
            self.CA3CA1weights = np.zeros(shape=(N, X, X))
        else:
            self.CA3CA1weights = initialCA3CA1weights

        if initialSemantics is None:
            if self.numOfSurfFeatures is not None:
                self.generateSemantics(N, X, self.numOfSurfFeatures)
            else:
                raise NameError("If semantics are not supplied the we must know the number of "
                                "surf features to generate semantics!")
        else:
            self.semanticIndices = initialSemantics

        # print("ECDG:\n%s\nCA3CA1:\n%s" % (self.ECDGweights, self.CA3CA1weights))


    def getOriginalValues(self, thresholdedFeatureVector):
        # Use the (sparse) feature vector (which has been dumb decoded so we know which input features should
        # in fact be active and the semantics we derived originally to get the input values that should be active
        # Decode using same semantics originally chosen
        activeSURFIndices = np.array(self.semanticIndices[thresholdedFeatureVector])
        decoded = np.zeros((self.numOfSurfFeatures,), dtype=np.int8)  # THIS COULD BE WRONG
        decoded[activeSURFIndices] = 1
        return decoded

    def getSemanticValues(self, featureVector):
        # Use the semantic indices decided upon initialisation of this DGState what the feature vector should look like.
        return np.array(featureVector[self.semanticIndices])

    def generateSemantics(self, N, X, numOfFeatures):
        # This one avoids duplicate SURF features being used in the same block
        self.semanticIndices = np.zeros((N,X), np.int8)
        for blockInd, block in enumerate(self.semanticIndices):
            self.semanticIndices[blockInd] = random.sample(range(numOfFeatures), X)
            # Jack Stevenson: xrange does not exist in python3

    def encode(self, inputActivationValues):
        # Dot product the ECDGweights with their activation to give activation values between -1 and 1
        outputActivationValues = np.zeros(shape=inputActivationValues.shape)

        # A block is a page of a 3d matrix
        for blocknum, block in enumerate(self.ECDGweights):
            outputActivationValues[blocknum] = np.dot(block, inputActivationValues[blocknum])

        # Output activations have the form [page1[ outputactivation0, ouputactivation1, outputactivation2],
        #                                   page2[ outputactivation0, outputactivation1, outputactivation2]]
        #                                   i.e columns are the output units, rows are the blocks
        # print("ouputActivation after:\n%s" % np.around(outputActivationValues, 3))
        
        # Smart collapse the whole matrix to get the winner
        encodedValues = smartCollapseMatrix(outputActivationValues)
        return encodedValues

    def decode(self, probabilitiesOfFiring):
        # Use "grey values" coming out of boltzmann to calculate the winners using smart collapse
        # These are the probabilities that the OUTPUT units of the sparse repreentation are on., since only one can be
        # on at a time we do a smart collapse
        probsReshaped = probabilitiesOfFiring.reshape(self.N,self.X)
        # print("Probabilities reshaped:\n%s" % probsReshaped)
        winningNeurons = smartCollapseMatrix(probsReshaped)
        probabilityOfActivation = np.zeros(winningNeurons.shape)

        # Apply transpose of ECDGweights to reverse the effect (i.e. calculate which inputs should be on given that an
        # output is on
        # for blocknum, block in enumerate(self.ECDGweights):
        for blocknum, block in enumerate(self.CA3CA1weights):
            # print("transpose ECDGweights:\n%s" % np.transpose(block))
            # print("winning output neurons:\n%s" % winningNeurons[blocknum])
            # We transpose the matrix as this allows us to see, given that output X is on, what are the probabilites
            # that input units A,B,C are on

            # TODO: Instead of using transpose of original ECDGweights, use the ECDGweights learnt by perceptron
            # probabilityOfActivation[blocknum] = np.dot(np.transpose(block), winningNeurons[blocknum])
            probabilityOfActivation[blocknum] = np.dot(block, winningNeurons[blocknum])

        # print("Probability of activation after ECDGweights have been applied:\n%s" % probabilityOfActivation)
        # We now have the probability that each feature is present, dumb decode it, i.e. if its still more than 50%
        # likely to be on, then count it as on
        thresholded = (probabilityOfActivation>=0.5)
        
        # Decode using same semantics originally chosen
        decoded = self.getOriginalValues(thresholded)
        return decoded

    def setECDGWeights(self, ECDGweights):
        self.ECDGweights = ECDGweights

    def setCA3CA1Weights(self, CA3CA1weights):
        self.CA3CA1weights = CA3CA1weights

    def learn(self, inputActivationValues, learnCA3CA1weights=False, learningrate=0.01):
        # Get semantic values for input
        sv = self.getSemanticValues(inputActivationValues)
        
        # Winning neurons are the DG output
        winningNeurons = self.encode(sv)

        # We only want to learn one set of weights at a time
        if learnCA3CA1weights:
            self.learnCA3CA1weights(inputActivationValues, winningNeurons, learningrate)
        else:
            self.learnECDGweights(winningNeurons, sv, learningrate)

    def learnECDGweights(self, winningNeurons, semanticValues, learningrate=0.01):
        """
        Winner takes all learning between EC and DG representations.
        inputActivationValue is the activation coming out of the EC, currently this is a boolean vector
        of whether the image has SURF features matching common ones discovered in the SURFExtraction phase.
        """

        # Give all none active neurons a negative activation to introduce negative ECDGweights
        # winningNeurons = (winningNeurons==0).choose(winningNeurons,-0.01)

        N = winningNeurons.shape[0]
        X = winningNeurons.shape[1]

        # This uses broadcasting to create a tiled transpose of winningNeurons,
        # Tiling converts a winning neuron activation (say neuron 2 won) [0, 0, 1] to the changes to be made to every
        # weight, I.e because neuron 2 one, all the connections to this neuron should be increased for this block, i.e.
        # [[0, 0, 0]
        #  [0, 0, 0]
        #  [1, 1, 1]] since rows are output units and columns input units in the weight representation

        # Otherwise have the winning output increase connections from all inputs to it
        # New axis required as otherwise broadcasting wont work, i.e. because its trying to broadcast (4,3) onto (4,3,1)
        self.ECDGweights += (learningrate*(winningNeurons.reshape(N,X,1)))*semanticValues[:, np.newaxis]

        # Normalise weights row by row (add up all elements of each row and divide each value by that number
        self.ECDGweights = normalise(self.ECDGweights,2)

    def learnCA3CA1weights(self, inputActivationValues, DGEncodedValues, learningrate=0.01):
        # Alter the encoded values given by the ECDGweights learnt going from EC-DG to account for the new data

        # We now know both the optimum output of the Boltzmann machine after winner take all has been done (after smart
        # collapse) - only if the data wasn't noisy in the first place? -
        # the collapsed output (sparse)
        # The optimum output of the boltzmann machine once smart collapse has been applied would be the original input
        # to it, if the data is clean.
        # 0 0 0 1 0 0 1 0 1 0 0 0
        # fully connected to the output which knows whether it should be on or off (the original data if trained with
        # clean data). If one is on and the other is on, increase the ECDGweights between them?

        # Ideally we would do offline learning? Be given a list of all the inputActivationValues, and all the
        # correctOutputActivationValues and get our error down below a threshold?
        """
        print("CA3CA1weights:\n%s" % self.CA3CA1weights)
        print("inputActivationValues:\n%s" % inputActivationValues)
        print("DGEncodedValues:\n%s" % DGEncodedValues)
        """
        
        # Threshold = bias?
        threshold = 0.5
        givenOutputPerBlock = np.zeros(shape=DGEncodedValues.shape)
        # thresholdedOutput at clipped at 0.5 is equivalent to bias?
        thresholdedOutput = np.zeros(shape=DGEncodedValues.shape, dtype=bool)

        # print("CA3CA1weights:\n%s" % np.around(self.CA3CA1weights, 3))
        for blocknum, block in enumerate(self.CA3CA1weights):
            # print("encoding CA1CA3 block:\n%s" % block)
            # print("input:\n%s" % inputActivationValues[blocknum])
            # print("OutputActivation:\n%s" % np.dot(block, DGEncodedValues[blocknum]))
            givenOutputPerBlock[blocknum] = np.dot(block, DGEncodedValues[blocknum])
            # givenOutputPerBlock needs to be changed back into a input vector by use of its semantics its encoded in
        thresholdedOutput = (givenOutputPerBlock>=threshold)

        # Bit of a hack, go from the calculated output of the CA3 representation to the CA1 representation:
        # Get the original values (giving the representation in CA1 form) by relating the CA3 to the semantics initially decided
        CA1Form = self.getOriginalValues(thresholdedOutput)
        #print("CA1Form:\n%s" % CA1Form)

        # Get the DG representation of this output (the decoded from boltzmann machine) so we can use the perceptron
        # learning rule on it (compare the desired output with the real output)
        realOutput = self.getSemanticValues(CA1Form)
        # Get the desired output by getting this form from the input activation (i.e. We know it if the decode was
        # perfect it should be the same as the original input
        desiredOutput = self.getSemanticValues(inputActivationValues)

        # Use Perceptron learning algorithm to change the weights in the direction of errors
        # NOTE: Should something be transposed here as the weights are from sparse to SURF-Features not the other way
        # round?
        difference = desiredOutput - realOutput

        N = difference.shape[0]
        X = difference.shape[1]
    
        # FIXME: Definitely not sure if DGEncodedValues is the right thing... just a guess
        changesInWeights = ((learningrate*(difference.reshape(N,X,1))*DGEncodedValues[:, np.newaxis]))

        self.CA3CA1weights += changesInWeights

        # Normalise
        # self.CA3CA1weights = normalise(self.CA3CA1weights,2)

class DGState:
    def __init__(self,  ec, dictGrids, dghelper=None, unittesting=0):
        self.dghelper = dghelper
        # HOOK:, needs to use EC data to define "combis" of features aswell

        if dghelper is not None:
            # Lets say for now that place whisker combos etc are all encoded normally, and SURF features are encoded
            # using WTA DG. In the end we may make sure that we have blocks referring only to location, blocks refering
            # only to whiskers, blocks refering only to light, etc.
            # FIXME: This needs changing when integrated to just get the number of surf features from ec!
            if unittesting:
                # Slice the SURF features from the numpy array
                self.numOfSurfFeatures = len(ec)
                self.surfFeatures = ec[-self.numOfSurfFeatures:]

            else:
                self.numOfSurfFeatures = len(ec.surfs)
                self.surfFeatures = ec.surfs 

            # Choose semantics by choosing X random features N times to make N blocks
            # For now be stupid, allow the same combinations to come up and the same indices to be compared with each
            # other for winner take all (will the conflict break it?)
            # Make this more intelligent later
            # Make random windows associated with the features, i.e. for N windows, choose X random features to encode,
            # make a matrix with the blocks and values
            #       <---X--->
            #    +-------------+
            # ^  | 0 0 0 0 1 0 |
            # |  | 1 0 0 0 0 0 |
            # N  |             |
            # |  |             |
            # |  |             |
            #    +-------------+


            self.semanticValues = dghelper.getSemanticValues(self.surfFeatures)

            # These are our input activations, once passed through a neural network with competitive learning applied to
            # its ECDGweights to encourage winner takes all, the output should only have 1 active value per block (row),
            # thus is sparse
            # What happens if none of the features are active?? Should the one with the highest weight win? Or should
            # there just be no activation in that block making it a even sparser matrix? I suspect the latter!
            self.encode()

        if not unittesting:
            if dghelper is None:
                self.encodedValues = np.array([])
            N_place = 13
            N_hd = 4       

            l=Location()       # NEW, pure place cells in DG
            l.setGrids(ec.grids, dictGrids)
            self.place=np.zeros(N_place)
            self.place[l.placeId] = 1

            self.hd_lightAhead = np.zeros(4)
            if ec.lightAhead == 1:
                self.hd_lightAhead = ec.hd.copy()

            self.whisker_combis = np.zeros(3)  # extract multi-whisker features.
            self.whisker_combis[0] = ec.whiskers[0] * ec.whiskers[1] * ec.whiskers[2]   # all on
            self.whisker_combis[1] = (1-ec.whiskers[0]) * (1-ec.whiskers[1]) * (1-ec.whiskers[2])   # none on
            self.whisker_combis[2] = ec.whiskers[0] * (1-ec.whiskers[1]) * ec.whiskers[2]   # both LR walls but no front

    def toVectorSurfOnly(self):
        if len(self.encodedValues) == 0:
            return self.encodedValues 
        else:
            return np.hstack((self.encodedValues.flatten()))

    def toVector(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead, self.whisker_combis, self.encodedValues.flatten()))

    def toVectorSensesOnly(self):
        return np.hstack((self.whisker_combis, self.toVectorSurfOnly()))
        # return np.hstack((self.whisker_combis, self.encodedValues.flatten()))

    def toVectorOdomOnly(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead))

    def smartCollapse(self):
        self.place = smartCollapse(self.place)

    def encode(self):
        self.encodedValues = self.dghelper.encode(self.semanticValues)
        
    def decode(self, probabilitiesOfFiring):
        self.decodedValues = self.dghelper.decode(probabilitiesOfFiring)
        return self.decodedValues


def smartCollapseMatrix(xs):
    # Use of argmax gives a maximum value no matter what, if a block is [0,0,0,0] the first index will be chosen as the maximum, this may not be desirable
    idx = np.argmax(xs, 1)
    r = np.zeros(xs.shape, np.int8)
    for row, col in enumerate(idx):
        r[row, col] = 1
    return r 


def smartCollapse(xs):
    idx=np.argmax(xs)
    r = np.zeros(xs.flatten().shape)
    r[idx]=1
    return r.reshape(xs.shape)


def addNoise(data, probability):
    noisyData = data.copy()
    for ind in range(len(data)):
        if random.random() < probability:
            noisyData[ind] = 1 - noisyData[ind]
    return noisyData


def accuracy(activation1, activation2):
    same = np.int8(np.logical_not(np.bitwise_xor(activation1, activation2)))
    return np.sum(same)/float(len(same))


def normalise(matrix, axis):
    rowsSummed = np.sum(matrix, axis)
    X = matrix.shape[1]
    N = matrix.shape[0]
    # If its a row normalisation (sum rows and divide by rows)
    if axis == 2:
        # print("rowsSummed:\n%s\nN:%d X:%d"% (rowsSummed, N,X))
        reshaped = np.reshape(rowsSummed, (N,X,1))
    elif axis == 1:
        reshaped = np.reshape(rowsSummed, (N,1,X))
    else:
        raise NameError("Axis must be rows or columns, axis == 2 is to add up a whole row and divide the row by that,\
            axis == 1 is to add up a whole column and divide the column by that")

    normalised = matrix / reshaped
    return normalised


def train_weights(trainingData, X, N, presentationOfData, learningrate=0.01):

    # Train the network on the clean initial data
    # initialECDGWeights = np.random.rand(N, X, X)/10
    # initialCA3CA1Weights = np.zeros(shape=(N, X, X))
    numOfSurfFeatures = len(trainingData[0])
    
    dgh = DGHelper(numOfSurfFeatures,X=X,N=N)
    
    # def __init__(self,  ec, dictGrids, dghelper):
    # trainingdg = DGState(trainingData[0], None, dgh)
    for x in range(presentationOfData):
        for data in trainingData:
            dgh.learn(data, False, learningrate)
    for x in range(presentationOfData):
        for data in trainingData:
            dgh.learn(data, True, learningrate)
    # Since the data has no patterns this might not work...

    return dgh


def calculate_performance(trainingData, inputDataSet, X, N, presentationOfData, learningrate=0.01):
    numOfImages = inputDataSet.shape[0]

    dgh = train_weights(trainingData, X, N, presentationOfData, learningrate)

    # Feed noisy data through EC-DG
    encodedData = np.zeros((numOfImages,N,X), dtype=np.int8)
    for imageNum, data in enumerate(inputDataSet):
        testingdg = DGState(data, None, dgh)
        encodedData[imageNum] = testingdg.encodedValues

    # pass DG onto CA1 as if it was the collapsed data,
    decodedData = np.zeros(inputDataSet.shape, dtype=np.int8)
    for imageNum, data in enumerate(encodedData):
        decodedData[imageNum] = dgh.decode(data)

    # Compare CA1 and non-noisy EC
    # Performance is an XNOR between the two codes before and after noise
    totalAccuracy = 0
    for imageNum, origData in enumerate(trainingData):
        totalAccuracy += accuracy(origData, decodedData[imageNum])
    totalAccuracy = totalAccuracy/float(inputDataSet.shape[0])*100

    # Calculate how much change the encode and decode has made (difference between noisy EC and CA1
    totalChange = 0
    for imageNum, noisyData in enumerate(inputDataSet):
        totalChange += accuracy(noisyData, decodedData[imageNum])
    totalChange = (1-(totalChange/float(inputDataSet.shape[0])))*100
    return totalAccuracy, totalChange

