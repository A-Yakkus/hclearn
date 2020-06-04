#!/usr/bin/env python
import os
import cv2
import numpy as np
import pyflann as flann
import unittest
import re
import makeMaze as mm
import pdb
from SURFExtractor import *


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.folder = rootFolder + "folderTest/"
        self.prefixFolder = rootFolder + "prefixFolderTest/"
        self.fileDict = {(0,'N'): [self.folder+"0/N/Dark.jpg"], \
                         (0,'S'): [self.folder+"0/S/Light.jpg"], \
                         (1,'E'): [self.folder+"1/E/Dark.jpg"], \
                         (2,'E'): [self.folder+"2/E/Dark.jpg", self.folder+"2/E/Day.jpg"], \
                         (2,'S'): [self.folder+"2/S/Night.jpg"]}
        self.pfileDict = {
                          ((3,4),'S'): [self.prefixFolder+"3-4-S.jpg"], \
                          ((2,3),'W'): [self.prefixFolder+"2-3-W-Dark.jpg", self.prefixFolder+"2-3-W-Light.jpg"], \
                          ((3,5),'S'): [self.prefixFolder+"3-5-S.jpg"], \
                          ((5,3),'E'): [self.prefixFolder+"5-3-E.jpg"], \
                          ((3,3),'N'): [self.prefixFolder+"3-3-N.jpg"], \
                          ((3,0),'S'): [self.prefixFolder+"3-0-S.jpg"], \
                          ((0,3),'E'): [self.prefixFolder+"0-3-E-Dark.jpg", self.prefixFolder+"0-3-E-Light.jpg"], \
                          }
        self.extractor = SURFExtractor(self.folder)

    def test_extractByPrefix(self):
        self.extractor.extractFilesByPrefix(self.prefixFolder)
        self.assertTrue(compareDicts(self.extractor.files, self.pfileDict))
        self.assertEqual(self.extractor.files, self.pfileDict)

    def test_extractByFolder(self):
        self.extractor.extractFilesByFolder(self.folder)
        self.assertTrue(compareDicts(self.extractor.files, self.fileDict))
        self.assertEqual(self.extractor.files, self.fileDict)

    def test_extractDescriptors(self):
        numOfFeaturesToExtract = 10
        #Difficult to test as I cant hand write descriptors!
        self.extractor.extractDescriptors(self.fileDict, numOfFeaturesToExtract)
        #Structure of dictionary should be
        #            im1d1 im1d2 im1d3 im1d4    im2d1 im2d2 im2d3 im2d4
        #{(2,'E'): [[sjefn,senjf,sefee,sjenfe],[sjenf,sefee,sjenf,sjenf]],
        #...
        #Less of a test more of a sanity check
        self.assertEqual(len(self.fileDict.keys()), len(self.extractor.descriptors.keys()))
        #Check length and type of descriptors
        descriptor0N = self.extractor.descriptors[(0,'N')]
        self.assertEqual(len(descriptor0N), 1)
        self.assertEqual(len(descriptor0N[0]), numOfFeaturesToExtract)
        self.assertEqual(len(descriptor0N[0][0]), 64)
        descriptor2E = self.extractor.descriptors[(2,'E')]
        self.assertEqual(len(descriptor2E), 2)
        self.assertEqual(len(descriptor2E[0]), numOfFeaturesToExtract)
        self.assertEqual(len(descriptor2E[0][0]), 64)
        self.assertEqual(len(descriptor2E[1]), numOfFeaturesToExtract)
        self.assertEqual(len(descriptor2E[1][5]), 64)

    def setupMerging(self):
        self.extractor.descriptors = {(0,'N'): [np.array([0,0], dtype=np.float32)], \
                                      (0,'S'): [np.array([0,1], dtype=np.float32)], \
                                      (1,'E'): [np.array([0,3], dtype=np.float32)], \
                                      (2,'E'): [np.array([1,1], dtype=np.float32), np.array([1.2, 1.2], dtype=np.float32)], \
                                      (2,'S'): [np.array([0,3.3], dtype=np.float32)]}

        #With a threshold of 0.3, we should get descriptors [0,0],[0,1],[0,3.15],[1.1,1.1] if all images are used, 
        #and the same but [1.1,1.1] being [1,1] if only the first images are used
        self.featuresDescDict = {(0,'N'): [np.array([0,0,0,1,0], dtype=np.int8)], \
                            (0,'S'): [np.array([0,0,1,0,0], dtype=np.int8)], \
                            (1,'E'): [np.array([1,0,0,0,0], dtype=np.int8)], \
                            (2,'E'): [np.array([0,0,0,0,1], dtype=np.int8), np.array([0,0,0,0,1], dtype=np.int8)], \
                            (2,'S'): [np.array([0,1,0,0,0], dtype=np.int8)]}

    def test_getFirstDescs(self):
        self.setupMerging()
        firstDescs = self.extractor.getFirstDescs()
        #Careful dictionaries arnt ordered...
        stackedDescs = np.array([[0,3],[0,3.3],[0,1],[0,0],[1,1]])
        #print("firstDescs:\n%s" % firstDescs)
        #print("stackedDescs:\n%s" % stackedDescs)
        self.assertTrue(np.allclose(stackedDescs, firstDescs))

    def test_mergeFeatures(self):
        self.setupMerging()

        self.extractor.mergeThreshold = 0.3
        self.extractor.mergeFeatures()
        #distance isn't just 0.3, it must be less than 0.1!
        correctMergedFeatures = np.array([[0,3.15],[0,1],[0,0],[1,1]])
        self.assertTrue(np.allclose(self.extractor.mergedFeatures, correctMergedFeatures))

        #[0.3] and [0,3.3] are no longer in range so should fail
        newCorrectMergedFeatures = np.array([[0,3],[0,3.3],[0,1],[0,0],[1,1]])
        self.extractor.mergeThreshold = 0.01
        self.extractor.mergeFeatures()
        self.assertTrue(np.allclose(self.extractor.mergedFeatures, newCorrectMergedFeatures))

    def test_generateFeatureVectors(self):
        self.setupMerging()

        self.extractor.mergedFeatures = self.extractor.getFirstDescs()

        self.extractor.matchThreshold = 0.3
        self.extractor.generateFeatureVectors()
        self.assertTrue(compareDicts(self.extractor.featuresDescDict, self.featuresDescDict))
        
        self.extractor.matchThreshold = 0.01
        self.extractor.generateFeatureVectors()
        #Should have been some failures due to matches not being made (match threshold too low so theyre not counted as a match)
        self.assertFalse(compareDicts(self.extractor.featuresDescDict, self.featuresDescDict))

    def test_generateFeatureRepresentations(self):
        self.setupMerging()
        #self.folder = self.prefixFolder

        newExtractor = SURFExtractor(self.folder)
        #FIX: If two photos are exactly the same, this will fail as they will be merged!
        newExtractor.mergeThreshold = 0 
        newExtractor.matchThreshold = 0.01
        newExtractor.generateFeatureRepresentations()

        """
        #Count how many images there are...
        numOfImages = 0
        for value in newExtractor.featuresDescDict.values():
            numOfImages += len(value)
        """
        #If we are using firstDesc, it only gets the first images of each key!
        numOfImages = len(newExtractor.featuresDescDict.keys())
        #Since none are merged, there should be numOfImages*numOfFeaturesForMerge
        numOfFeaturesWithoutMerge = newExtractor.maxFeaturesForMerging*numOfImages

        #Test this by getting first key values length of first image description (number of features in its vector)
        numOfFeaturesFirstKey =  len(newExtractor.featuresDescDict[newExtractor.featuresDescDict.keys()[0]][0])
        numOfFeaturesSecondKey =  len(newExtractor.featuresDescDict[newExtractor.featuresDescDict.keys()[1]][0])
        #Number of features going into merge
        self.assertEqual(numOfFeaturesFirstKey, numOfFeaturesWithoutMerge)
        self.assertEqual(numOfFeaturesSecondKey, numOfFeaturesWithoutMerge)
        
        newExtractor = SURFExtractor(self.folder)
        newExtractor.mergeThreshold = 0.05
        newExtractor.matchThreshold = 0.05
        newExtractor.generateFeatureRepresentations()

        #Test this by getting first key values length of first image description (number of features in its vector)
        numOfFeaturesFirstKey =  len(newExtractor.featuresDescDict[newExtractor.featuresDescDict.keys()[0]][0])
        numOfFeaturesSecondKey =  len(newExtractor.featuresDescDict[newExtractor.featuresDescDict.keys()[1]][0])
        #Number of features going into merge
        self.assertLess(numOfFeaturesFirstKey, numOfFeaturesWithoutMerge)
        self.assertLess(numOfFeaturesSecondKey, numOfFeaturesWithoutMerge)

    def test_mergeQuality(self):
        newExtractor = SURFExtractor(self.folder)
        newExtractor.mergeThreshold = 0.15
        newExtractor.matchThreshold = 0.35
        newExtractor.generateFeatureRepresentations()


    def test_merge_senses_and_features(self):
        N_mazeSize = 3
        [dictSenses, dictAvailableActions, dictNext] = mm.makeMaze(N_mazeSize)     #make maze, including ideal percepts at each place
        #print("TESTING\n%s"%dictSenses)

    def test_makeSURFRepresentation(self):
        #Should simply call the generate feature representation method!
        sdict = makeSURFRepresentation()

        #print("Extracted features per image:\n")
        #for key in sdict.keys():
        #    for featureVec in sdict[key]:
        #        print("image key %s, features:\n%s" % ((key),featureVec))

    def test_findSurfs(self):
        #Test that given an x y and direction from a dictionary it is possible to find the surf feature!
        adict = {((1,4),'N'): [np.array([1,0])], ((3,4),'E'): [np.array([1,1]), np.array([0,1])]}
        x = 3
        y = 4
        ith = 0
        sf34E = mm.findSurfs(x,y,ith,adict)
        self.assertTrue(np.all(sf34E == np.array([1,1])))

        x = 1
        y = 4
        ith = 1
        sf14N = mm.findSurfs(x,y,ith,adict)
        self.assertTrue(np.all(sf14N == np.array([1,0])))

        x = 1
        y = 3
        ith = 1
        sf13N = mm.findSurfs(x,y,ith,adict)
        self.assertTrue(np.all(sf13N == np.array([0,0])))
        #self.assertRaises(NameError, findSurfs, x,y,ith,adict)
    
    def test_Senses_init(self):
        #Create a Sense with a known dictionary and see if the results are right
        pass


class TestComparisons(unittest.TestCase):
    def setUp(self):
        self.regentImages = [prefixFolder + "3-1-S-Midday.jpg", prefixFolder + "4-3-E-Midday.jpg"]
        self.images= ["room1.jpg", "room2.jpg", "window.jpg", "labs1.jpg", "labs2.jpg", "bottle.jpg"]
        self.images = [ rootFolder + im for im in self.images ]
        #Load the images
        #Multiple versions of some as the SURF drawer draws on them, thus fucking up the next extraction
        self.imsMat = []
        for image in self.images:
            self.imsMat.append(cv.LoadImageM(image, cv.CV_LOAD_IMAGE_GRAYSCALE))

        self.imThresholded = np.zeros((len(self.images),len(self.images)))

        #Careful again you keep drawing on the same image!
        for x in range(len(self.images)):
            for y in range(len(self.images)):
                self.imThresholded[x][y] = computeDescriptorCloseness(self.imsMat[x], self.imsMat[y], 0)

    #@unittest.skip("Saving time whilst testing other")
    def test_same(self):
        #Matching with itself should be 1
        self.assertAlmostEqual(self.imThresholded[0][0], 1)

    #@unittest.skip("Saving time whilst testing other")
    def test_similar_greater_than_dissimilar(self):
        """
        for x in range(len(self.images)):
            for y in range(len(self.images)):
                print("Im%d%d number of close matches: %f" % (x,y,self.imThresholded[x][y]))
        """

        #rooms 1 and 2 are more similar to eachother than either room with the window
        self.assertGreater(self.imThresholded[0][1], self.imThresholded[0][2])
        self.assertGreater(self.imThresholded[0][1], self.imThresholded[1][2])
        #labs 1 and 2 are more similar to eachother than either lab with the window
        self.assertLess(self.imThresholded[1][2], self.imThresholded[3][4])
        self.assertLess(self.imThresholded[1][2], self.imThresholded[4][3])

    #@unittest.skip("Saving time whilst testing other")
    def test_drawing(self):
        im1DrawnOn = cv.LoadImageM(self.regentImages[0], cv.CV_LOAD_IMAGE_GRAYSCALE)
        im2DrawnOn = cv.LoadImageM(self.regentImages[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        extractSURFFeatures(im1DrawnOn, 1, 10)
        extractSURFFeatures(im2DrawnOn, 1, 10) 

        cv.NamedWindow("SURFFeatures 1", cv.CV_WINDOW_AUTOSIZE)
        cv.ShowImage("im1DrawnOn", im1DrawnOn)
        cv.NamedWindow("SURFFeatures 2", cv.CV_WINDOW_AUTOSIZE)
        cv.ShowImage("im2DrawnOn", im2DrawnOn)
        cv.WaitKey(0)
        self.assertTrue(True)

    #@unittest.skip("Saving time whilst testing other")
    def test_featureMerging(self):
        feature1=[1,2,3]
        feature2=[2,3,4]
        newFeature = mergeFeaturePair(feature1,feature2) 
        expectedFeature=[1.5, 2.5, 3.5]
        for i in range(2):
            self.assertAlmostEqual(newFeature[i], expectedFeature[i])

    #@unittest.skip("Saving time whilst testing other")
    def mergeSetup(self):
        self.imsDescs = [0]*len(self.images)
        for im in range(len(self.images)):
            self.imsDescs[im] = extractSURFFeatures(self.imsMat[im], 0)

        self.allFeatures = np.vstack((self.imsDescs[0], self.imsDescs[1], self.imsDescs[2], self.imsDescs[3], self.imsDescs[5]))

        #mergeThreshold=0.5
        mergeThreshold=0.08
        self.mergedFeatures = mergeFeatures(self.allFeatures, mergeThreshold)

    #@unittest.skip("Saving time whilst testing other")
    def test_mergeFeatureSets(self):
        self.mergeSetup()
        beforeFeatureCount = np.shape(self.allFeatures)[0]
        afterFeatureCount = len(self.mergedFeatures)
        #ON print("Had %d features, now have %d feature" % (beforeFeatureCount, afterFeatureCount))
        self.assertGreater(beforeFeatureCount, afterFeatureCount)

    #@unittest.skip("Saving time whilst testing other")
    def test_overlap_of_feature_matches(self):
        self.mergeSetup()

        #matchThreshold=0.4
        matchThreshold=0.1

        imFeaturesWithinThreshold = [0]*len(self.images)
        newPercentageDifferenceims = [0]*len(self.images)
        imFeatureVector = [0]*len(self.images)
        for i in range(len(self.images)):
            #Calculate the features that are within the threshold of the merged feature set
            newPercentageDifferenceims[i], imFeaturesWithinThreshold[i], averageDistance  = compareDescriptors(self.mergedFeatures, self. imsDescs[i], matchThreshold)
            #Make these features into a boolean feature vector
            imFeatureVector[i] = findBooleanFeatureVector(len(self.mergedFeatures), imFeaturesWithinThreshold[i])
            #ON print("image has %d active features in the merged feature vector:\n %s" % (sum(imFeatureVector[i]), imFeatureVector[i]))
            #Each vector should have atleast one feature active
            self.assertGreater(sum(imFeatureVector[i]), 0)

        #Make a matrix of how features overlap
        overlap = np.ndarray((len(self.images), len(self.images)), np.object)
        for compareImage in range(len(self.images)):
            overlap[0][compareImage] = calculateSharedFeatures(imFeatureVector[0], imFeatureVector[compareImage])
            #ON print("overlap %d%d shares: %d features, feature vector:\n %s \n\n\n\n"  % (0, compareImage, sum(overlap[0][compareImage]), overlap[0][compareImage]))

        #self.assertGreater(sum(overlap[0][1]), sum(overlap[0][2]))

        #If the featureVectors are exclusive, the superset of one vector shouldn't be any of the others, this should be the case for ones made more compact
        nzIndices = [0]*len(imFeatureVector)
        for featureVInd in range(len(imFeatureVector)):
            nzIndices[featureVInd] = set(np.nonzero(imFeatureVector[featureVInd])[0])

        supersets = [[False]*len(imFeatureVector)]*len(imFeatureVector)
        #print supersets
        for x in range(len(imFeatureVector)):
            for y in range(len(imFeatureVector)):
                if x != y:
                    supersets[x][y] = nzIndices[x].issuperset(nzIndices[y])
        #flatten the list of lists
        supersetF = [superset for sublist in supersets for superset in sublist]

        #Unless the same picture is shown twice, no set of features should be a superset of any others
        #Is this the case or should they just not have the same features active?
        self.assertFalse(any(supersetF), "One feature vector is a superset of the other, thus one image cannot be uniquely described")

    #@unittest.skip("Doesnt work yet")
    def test_recognition(self):
        #Given six images, each with their own location (1,2 or 3) train on three and merge features
        #[(image1, loc1), (image2, loc2), (image3, loc3), (image4, loc1), (image5, loc2), (image6, loc3)]
        imagesCat = [('room1.jpg',1), ('room2.jpg',1), ('labs1.jpg', 2), ('labs2.jpg', 2), ('outside1.jpg', 3), ('outside2.jpg', 3)]#, ('labs3.jpg', 2)]
        imagesCat = [ (rootFolder+tup[0], tup[1]) for tup in imagesCat ]

    
        #Calculate closeness for remaining features
        imsStored = [0, 2, 4]
        imsTesting = [1, 3, 5]#, 6]

        imsCatMat = []
        for image in imagesCat:
            imsCatMat.append(cv.LoadImageM(image[0], cv.CV_LOAD_IMAGE_GRAYSCALE))

        #With the remaining three images, see if their location can be determined purely from the number of matches between them and the three images
        imsCatDescs = [0]*len(imsCatMat)
        #Could use more features
        for im in range(len(imsCatMat)):
            if im in imsStored:
                #ON print("%d is stored" % im)
                imsCatDescs[im] = extractSURFFeatures(imsCatMat[im], 0)
            else:
                #ON print("%d is testing" % im)
                imsCatDescs[im] = extractSURFFeatures(imsCatMat[im], 0, 10)
            #ON print("%d has %d features" % (im, len(imsCatDescs[im])))

        allCatFeatures = np.vstack((imsCatDescs[0], imsCatDescs[2], imsCatDescs[4]))

        mergeThreshold=0.1
        mergedCatFeatures = mergeFeatures(allCatFeatures, mergeThreshold)
        f = trainFLANN(mergedCatFeatures)
        matchThreshold=0.1

        featureVecs = np.ndarray((len(imsCatMat),len(mergedCatFeatures)), np.int8)
        for i in range(len(imsCatMat)):
            featureVecs[i,:] = calculateFeatureVector(f, imsCatDescs[i], matchThreshold, len(mergedCatFeatures))
        #ON print("Feature vectors for recognition are:\n%s" % featureVecs)
        #print featureVecs
        
        closestImages = []
        #print (featureVecs)
        for im in imsTesting:
            #closestImage = (indexOfImage, featuresInCommon)
            closestImage = (-1, -1)
            for testIm in imsStored:
                overlap = np.sum(calculateSharedFeatures(featureVecs[im,:], featureVecs[testIm,:]))
                #ON print("im %d and im %d overlap %d features" % (im, testIm, overlap))
                if overlap > closestImage[1]:
                    closestImage = (testIm, overlap)
            print("closest image ind %d, with %d overlapping features" % (closestImage[0], closestImage[1]))
            closestImages.append((imagesCat[im][1], imagesCat[closestImage[0]][1]))
        #ON print closestImages
        similarity = [ loc1 == loc2 for (loc1, loc2) in closestImages ]
        #print (similarity)
        #The classifier is correct if the locations match for all image pairs
        self.assertTrue(all(similarity))
        #ON print("\n\n\n\n")


if __name__ == '__main__':
    #unittest.main()
    comparisonsSuite = unittest.TestLoader().loadTestsFromTestCase(TestComparisons)
    extractSuite  = unittest.TestLoader().loadTestsFromTestCase(TestExtractor)
    #allSuites = [extractSuite]
    #allSuites = [extractSuite, comparisonsSuite]
    allSuites = comparisonsSuite
    completeSuite = unittest.TestSuite(allSuites)
    unittest.TextTestRunner(verbosity=2).run(completeSuite)

    """
    #suite.addTest(TestComparisons(unittest
    #unittest.getTestCaseNames(testCaseThing)
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestComparisons)
    unittest.TextTestRunner(verbosity=2).run(suite)

    class (unittest.TestCase):
    class TestExtractor(unittest.TestCase):
    """
