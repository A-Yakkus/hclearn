#!/usr/bin/env python
import os
import cv2
import numpy as np
import pyflann as flann
import unittest
import re
#import workspace.src.main.makeMaze as mm
import pdb

#rootFolder = "/Users/alansaul/Work/CompSci/SURE/hclearn_alan/"
rootFolder = os.getcwd()+"/"   # "/home/charles/git/hclearn/"

#This is the folder being used by makeSURFRepresentation to create the surf features for learnWeights
#prefixFolder = rootFolder + "../src/res/DCSCourtyard/"
prefixFolder = rootFolder + "../res/DCSCourtyard/"

class SURFExtractor(object):
    directions = ['N','E','S','W']
    #Stores the featuresDict, filesDict and featuresDescDict

    #Given a folder it will extract descriptors, merge them and generate the featureVectors for each image, and store them

    #Requires the folder name
    def __init__(self, folderName, maxFeaturesForMerging=15, maxFeaturesForMatching=20, mergeThreshold=0.15, matchThreshold=0.2):
        self.folder = folderName
        self.maxFeaturesForMerging = maxFeaturesForMerging
        self.maxFeaturesForMatching = maxFeaturesForMatching
        self.mergeThreshold = mergeThreshold
        self.matchThreshold = matchThreshold

    #Extract files by name prefix, store in dictionary
    def extractFilesByPrefix(self, folder):
        self.files = {}
        #Key should be of form ((x,y),dir)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                parts = re.split("[-,\.]", file)
                #Test that it is (NUM-NUM-DIRECTION-whatever)
                if len(parts)>=2 and parts[0].isdigit() and parts[0].isdigit() and (parts[2][0].isalpha and len(parts[2]) == 1):
                    if parts[2][0] in self.directions:
                        key = ((int(parts[0]), int(parts[1])),parts[2])
                        #If it doesnt already exist, make this key
                        if key not in self.files.keys():
                            self.files[key] = []
                        fullFilePath = os.path.join(folder,file)
                        #Add the new file onto the end of the keys list (since there can be multiple images for one direction)
                        self.files[key].append(fullFilePath)
                    else:
                        raise NameError("Heading is: %s\nit should be N S E or W" % parts[2])
                else:
                    pass
                    #print (folder)
                    #print (file)
                    #raise NameError("File: %s\ndoes not fit naming convention INT-INT-HEADING" % file)
        else:
            raise NameError("Folder does not exists")

    #Extract files by folders subfolders, store in dictionary
    def extractFilesByFolder(self, folder):
        #for subdir in os.listdir(folder):
        self.files = {}
        #Since each location is named after a number we can do this
        locCount = 0
        #For each location incrementally (locations cannot have gaps!) check to see if it exists
        while os.path.exists(os.path.join(folder,str(locCount))):
            #If it does exist then make this path
            locsubdir = os.path.join(folder,str(locCount))
            #For each direction (N,E,S,W) check to see if the subdirectory exists and whether it has any images in it
            for direction in self.directions:
                dirsubdir = os.path.join(locsubdir,direction)
                if os.path.exists(dirsubdir) and (len(os.listdir(dirsubdir)) > 0):
                    self.files[(locCount,direction)] = []
                    #For each file in the direction subdirectory, add a tuple key (locationNum, 'Direction') with the file as the value
                    for file in os.listdir(dirsubdir):
                        fullFilePath = os.path.join(dirsubdir,file)
                        self.files[(locCount,direction)].append(fullFilePath)
            #Go onto the next location, (they must run incrementally)
            locCount += 1

    #Extract maxFeaturesPerImage best descriptors per image and store in dictionary
    def extractDescriptors(self, files, maxNumOfDescriptors):
        self.descriptors = {}
        for (loc, dir) in files.keys():
            #print("\n%d, %s key" % (loc,dir))
            self.descriptors[(loc,dir)] = []
            for image in files[(loc,dir)]:
                cvIm = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                imFeatures = extractSURFFeatures(cvIm,0,maxNumOfDescriptors)
                self.descriptors[(loc,dir)].append(imFeatures)

    #Use the descriptors recently extracted to merge them down into a subset describing lots of them
    def mergeFeatures(self, random=0):
        #Make a subset of all the descriptors
        if random:
            #TODO: Add random selection of descriptors rather than just first images of each direction
            #Randomly choose 'random' features from the dictionary
            pass
        else:
            #Take the first picture from each loc, dir pair
            imageDescs = self.getFirstDescs()

        #print("Before:\n%d,%d" % (imageDescs.shape))
        #merge the features within range
        self.mergedFeatures = mergeFeatures(imageDescs, self.mergeThreshold)
        #print("After:\n%d,%d" % (self.mergedFeatures.shape))

    #Get the all of the descriptors for the first image
    def getFirstDescs(self):
        #Check that the dictionary isn't empty...
        if self.descriptors:
            #Do something more pythonic?
            first = True
            for key in self.descriptors.keys():
                if len(self.descriptors[key]) > 0:
                    #If the features already exists, append the new descriptors on, else initialise it 
                    if first:
                        #Get the first images descriptors in this list of several images descriptors
                        features = self.descriptors[key][0]
                        first = False
                    else:
                        features = np.vstack((features, self.descriptors[key][0]))
            return features
        else:
            raise NameError('Dictionary of descriptors is empty, there should be items in it in order to get the first ones!')

    #Go through each location, then each picture, and generate their boolean feature vector
    def generateFeatureVectors(self):
        #Given all the real descriptors, calculate their boolean vector form
        self.flann = trainFLANN(self.mergedFeatures)
        numOfFeatures = len(self.mergedFeatures)
        self.featuresDescDict = {}
        for (loc, dir) in self.descriptors.keys():
            self.featuresDescDict[(loc,dir)] = []
            for imageDescs in self.descriptors[(loc,dir)]:
                #Calculate the feature vector for this image and add it to the list for this key
                featureVec = calculateFeatureVector(self.flann, imageDescs, self.matchThreshold, numOfFeatures)
                self.featuresDescDict[(loc,dir)].append(featureVec)

    #Generate featureVectors
    def generateFeatureRepresentations(self, byFolder=1):
        if self.folder:
            print(self.folder)
            #First extract the files
            if byFolder:
                self.extractFilesByFolder(self.folder)
            else:
                self.extractFilesByPrefix(self.folder)

            #Extract the descriptors of all top X features
            self.extractDescriptors(self.files, self.maxFeaturesForMerging)
            
            #print("Descriptors before merge:\n%s" % self.descriptors)
            #print("Merge threshold:\n%s" % self.mergeThreshold)
            #Merge features so that we only have a small subset which are used to describe each image 
            #Unless otherwise stated this will use the FIRST image of each direction ONLY to train with
            self.mergeFeatures()
            #print("Merged features:\n%s" % self.mergedFeatures)

            #Generate features for each image
            #Get more features than we used for merging to be used for matching:
            self.extractDescriptors(self.files, self.maxFeaturesForMatching)

            self.generateFeatureVectors()
        else:
            raise NameError("Folder to select features from has not been provided")




def makeSURFRepresentation():
    #Make all things SURFY (its dictionary) and give back to makeMaze
    se = SURFExtractor(prefixFolder, matchThreshold=0.5)
    #FIX: If two photos are exactly the same, this will fail as they will be merged!
    se.mergeThreshold = 0.06 #0.05
    se.matchThreshold = 0.2
    se.generateFeatureRepresentations(0)
    #print("FEATUREDESCDICT: %s" % se.featuresDescDict)
    return se.featuresDescDict

def compareDicts(dict1,dict2):
    correctMatches=[]
    for key in dict1:
        for itemnum, image in enumerate(dict1[key]):
            correctMatches.append(np.all(image == dict2[key][itemnum]))
    return np.all(np.array(correctMatches))


def extractSURFFeatures(image,draw, N=7):
    #Extract SURF features (between 10000 and 30000 are good values)
    #(keypoints, descriptors) = cv2.ExtractSURF(image, None, cv2.CreateMemStorage(), (0, 100, 3, 2) )

#upgrade to opencv2:
    #surf = cv2.SURF(400)
    surf = cv2.xfeatures2d.SURF_create(400)
    (keypoints, descriptors)= surf.detectAndCompute(image,None)

    #pdb.set_trace()

    #Want to take the X best ones
    # original key function -> (lambda keypoint,descriptors: keypoint.size)

    sortedDescriptorListPairs = [descriptor for keypoint, descriptor in sorted(zip(keypoints, descriptors), key=(lambda x: x[0].size), reverse=True) ]
    #np.array(sortedDescriptorListPairs[0:N])
    #print("Num of keypoints: %d  Num of descriptors: %d " % (len(keypoints), len(descriptors)))
    if draw:
        for ((x, y), laplacian, size, dir, hessian) in keypoints:
            #print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)
            #For each feature draw a circle around it
            #Careful! Drawing on the images changes the images!!!
            cv2.Circle(image, (int(x),int(y)), size, (255.0, 0.0, 0.0,  0.0), 2)
    #return np.array(descriptors)
    return np.array(sortedDescriptorListPairs[0:N])

def computeDescriptorCloseness(image1,image2,draw):
    image2Descs = extractSURFFeatures(image2,draw)
    image1Descs = extractSURFFeatures(image1,draw)
    return compareDescriptors(image1Descs, image2Descs)[0]

def compareDescriptors(indexDescriptors, imageDescs, threshold=0.05):
    #print(image1Descs[0])
    resultInds, distances = findClosestMatchingFeaturesPairs(indexDescriptors, 1, imageDescs)

    #Calculate the average distance from a feature in the test to its nearest neighbor in the training data
    averageDistance = np.average(distances)
    featuresWithinThreshold = findMatchingFeatures(resultInds, distances, threshold)

    #print("Average distance: %f" % averageDistance)
    #Out of how many matches that could be found (image2Descs is out test data, indexDescriptors is our training data)
    #How many matches were below the threshold distance?
    #print("%d matches out of %d potential matches" % (len(featuresWithinThreshold), len(indexDescriptors)))

    #Get unique values (shouldnt be used multiple times?
    #uniqueMatches = set([val for (x, val, y) in featuresWithinThreshold]) 
    #print [val for (x, val, y) in featuresWithinThreshold] #uniqueMatchesd
    #Bit of a hack but seems to work, just because you only have 2 features, doesnt mean if those two are correct you are a perfect match
    #Getting 98 out of 100 should be better than 2 out of 2
    percentageClose = 2*(float(len(featuresWithinThreshold))/(len(imageDescs)+len(indexDescriptors)))
    #percentageClose = 2*(float(len(featuresWithinThreshold))/(len(indexDescriptors)))
    return percentageClose, featuresWithinThreshold, averageDistance

def findClosestMatchingFeaturesPairs(trainingData, k, testData=None):
    duplicatesIncluded = False
    if testData == None:
        #If we are using the training data as the test data, there will always be a distance of 0 without having k>1
        assert(k>1)
        duplicatesIncluded = True
        testData = trainingData

    #Set up a FLANN classifier (Fast Approximate Nearest Neighbor)
    f=trainFLANN(trainingData)
    #print("Params used to find nearest neighbours: ", params)
    #Try and match all the features found in the second image with those stored in the k nearest neighbor
    results, dists = f.nn_index(testData, k)
    #print("Distances to nearest neighbour: ", dists)
    #If we are using the trainingdata as the test data then there will always be a closest match
    if duplicatesIncluded:
        dists = dists[:,1:]
        results = results[:,1:]
        #results, dists =  results[:][1:], dists[:][1:]
    return results, dists 

def trainFLANN(trainingData):
    #Set up a FLANN classifier (Fast Approximate Nearest Neighbor)
    f = flann.FLANN()
    #Set the first image as the base (in the future this will be the "generalised" feature matrix
    f.build_index(trainingData)
    return f

def findNearestFeatures(flann, featureDescs):
    results, dists = flann.nn_index(featureDescs, 1)
    return results, dists

def calculateFeatureVector(flann, featureDescs, matchThreshold, sizeOfFeatureVector):
    results, dists = findNearestFeatures(flann, featureDescs)
    #FIX: THIS IS WHERE THE PROBLEM IS! ITS CUTTING OFF THINGS BELOW A THRESHOLD THUS THEY NO LONGER EXIST
    featuresWithinThreshold = findMatchingFeatures(results, dists, matchThreshold)
    featureVector = findBooleanFeatureVector(sizeOfFeatureVector, featuresWithinThreshold)
    return featureVector

#Find features which are within a threshold
def findMatchingFeatures(resultInds, dists, threshold):
    #Only take into account matches above a certain threshold
    #Need a list of features which match, including the indices of the training data being merged, the training data indices being merged, and the distance between them for reference
    #FIX: THIS IS WHERE THE PROBLEM IS! ITS CUTTING OFF THINGS BELOW A THRESHOLD THUS THEY NO LONGER EXIST
    featuresWithinThreshold = [(featureInd, int(closestInd), float(dist)) for featureInd, closestInd, dist in zip(range(len(resultInds)), resultInds, dists) if abs(dist) < threshold]
    sortedFeaturesWithinThreshold = sorted(featuresWithinThreshold, key=lambda feature: feature[2])
    #print("Thresholded has %(number)d pairs within the threshold\n" % {"number": len(featuresWithinThreshold)})
    return sortedFeaturesWithinThreshold

def mergeFeatures(trainingSet, threshold=0.05):
    finishedMerging = False
    while not finishedMerging:
        #Generate distance to nearest features in test data (all other images features)
        resultInds, distances = findClosestMatchingFeaturesPairs(trainingSet, 2)

        #Get the subset of matches such that distance < tolerance
        featuresWithinThreshold = findMatchingFeatures(resultInds, distances, threshold)

        #Strip distances which are equal to 0 as they are effectively already merged
        featuresWithinThreshold = [(featureInd, closestInd, dist) for featureInd, closestInd, dist in featuresWithinThreshold if abs(dist) > 0]

        #print("Average distance between features: %f" % averageDist)
        #ON print("%d features still need merging" % len(featuresWithinThreshold))
        #ON print("trainingSet length: %d" % len(trainingSet))

        #If subset is > 0
        if len(featuresWithinThreshold) > 0:
            #ON print "attempting to merge"
            newTrainingSet = [] 
            trainingIndicesMerged = set()

            #Merge the matches and add to a new training set
            for (featureInd, closestFeatureInd, distance) in featuresWithinThreshold:
                #Can only merge with one feature at a time! Since they are ordered closest features get preference
                if (featureInd not in trainingIndicesMerged) and (closestFeatureInd not in trainingIndicesMerged):
                    #ON print("FeatureInd: %d closestFeatureInd %d distance: %f" % (featureInd, closestFeatureInd, distance))
                    newGeneralFeature = mergeFeaturePair(trainingSet[featureInd], trainingSet[closestFeatureInd])
                    newTrainingSet.append(list(newGeneralFeature))
                    #newTrainingSet = newTrainingSet.vstack((newTrainingSet, newGeneralFeature))

                    #print("Merged feature:")
                    #print newGeneralFeature
                    #ON print("New training set length: %d" % len(newTrainingSet))
                    #Keep track of the indices merged so we can remove them later
                    trainingIndicesMerged.add(featureInd)
                    trainingIndicesMerged.add(closestFeatureInd)

            #Add the remaining from the test and training set to a new test set
            allTrainingIndices = set(range(len(trainingSet)))
            trainingIndicesReused =  allTrainingIndices.difference(trainingIndicesMerged)

            newTrainingSet = np.concatenate((np.array(newTrainingSet), np.array(trainingSet[list(trainingIndicesReused)])))

            #ON print("training set length at end of iteration: %d" % len(newTrainingSet))

            trainingSet = newTrainingSet
        #Else there is no more to merge so we are finished
        else:
            finishedMerging = True
    #ON print("final training set length: %d" % len(trainingSet))
    return trainingSet

def mergeFeaturePair(feature1, feature2):
    #Merge the feature by combining their descriptors
    newFeature = (np.array(feature1) + np.array(feature2)) / float(2)
    return newFeature

def findBooleanFeatureVector(totalNumberOfFeatures, featuresWithinThreshold):
    #Make a feature vector
    featureVector = np.zeros(totalNumberOfFeatures, np.int8) 

    #Get all closest indexes, remove duplicates to save time
    indices = set(([feature[1] for feature in featuresWithinThreshold]))
    #print("indices being accessed: %s" % indices)
    #Set the indices in the feature vector as on
    if len(indices) > 0:
        featureVector[list(indices)] = 1
    return featureVector

def calculateSharedFeatures(featureVector1, featureVector2):
    sharedVector = np.bitwise_and(featureVector1, featureVector2)
    return sharedVector


