import numpy as np
import matplotlib.pyplot as plt
import os

# A function to return the index
# of a minimum element in an array
def selectMin(dist):
    return dist.index(min(dist))

# A function to return the index
# of a maximum element in an array
def selectMax(dist):
    return dist.index(max(dist))

# Function to read our label's data
# given it's directory path
def loadLabels(dirpath):
    return np.loadtxt(dirpath, dtype=int)

# a Function to calculate the Euclidean
# distance between to points a, b
# or two vectors or two matrices.
def euclidean(a, b):
    return np.linalg.norm(a-b)

# This function checks wether an element exists in a
# given array
def check(element,array):
    return any(np.array_equal(x, element) for x in array)

# This function transforms an input image
# to a grayscaled image given a threshold
def grayScaleToBinary(image, threshold):
    imageGray = image
    imageGray[imageGray>threshold] = 1
    imageGray[imageGray!=1] = 0
    return imageGray

#
def loadData(dirName):
    #2400
    imageCount = len(os.listdir(dirName))-1
    #784
    imageSize = len(plt.imread(dirName+'/1.jpg').flatten())
    # Initialize array of input images
    xN = np.zeros((imageCount, imageSize), dtype=int)

    # Assign each index of xN an image from the directory
    for i in range(1, imageCount+1):
        currentImage = plt.imread(dirName+'/'+ str(i) +'.jpg').flatten()
        xN[i-1] = grayScaleToBinary(currentImage, 140)
    # Read labels from text file
    labels = loadLabels(dirName+'/Training Labels.txt')
    return xN, labels

# Initialize Centroids array given the number of clusters.
def initializeCentroids(xN, clusters):
    # Initialize an array of centroids 10x784
    centroids = np.zeros((clusters, xN.shape[1]))
    # Pick random number
    rand = np.random.randint(0, 2400)
    # Assign the first centroid as xN[index]
    centroids[0] = xN[rand]
    currCentroid = xN[rand]
    
    # Iterate for K times
    for i in range(1, centroids.shape[0]):
        maximum = -1
        maxIndex = currCentroid
        
        # Calculate the Euclidean Distance
        # and assign the farthest distance
        # to be the new centroid
        for j in range(xN.shape[0]):
            distance = euclidean(currCentroid, xN[j])
            if distance > maximum:
                if not check(xN[j], centroids):
                    maximum = distance
                    maxIndex = xN[j]
        
        currCentroid = maxIndex
        centroids[i] = maxIndex
    return centroids

def findMinimumMembership(point, centroids):
    minimum = np.inf
    minimumIndex = -1

    for i in range(centroids.shape[0]):
        distance = euclidean(point, centroids[i])

        if distance < minimum:
            minimum = distance
            minimumIndex = i

    return minimumIndex

# This function creates a Membership array
# in which each index corresponds to the cluster
# this image belongs to
def returnMembershipArray(xN, centroids):
    # Initialize membership array of 0s
    membership = np.zeros((xN.shape[0], centroids.shape[0]))

    # Fill in membership array with
    # centroid corresponding to the
    # minimum index after Euclidean distance
    # calculation
    for i in range(0, xN.shape[0]):
        currentImage = xN[i]
        minimumDistance = findMinimumMembership(currentImage, centroids)
        membership[i][minimumDistance] = 1

    return membership

# Update the old centroids
# and returning a new set of centroids
def updateCentroids(x, membership, centroids, clusters):
    tRanks = np.transpose(membership)
    newCentroids = np.zeros((clusters, x.shape[1]))
    
    for i in range(tRanks.shape[0]):
        sum = np.zeros((x.shape[1]));
        count = 0;
        tRank = tRanks[i]
        
        for j in range(x.shape[0]):
            if tRank[j]==1:
                count += 1
                sum = np.add(x[j], sum)
        
        newCentroids[i] = sum/count
    
    return newCentroids

# Find the total distance between images
# and their corresponding centroids
def totalDistance(xN, centroids):
    membership = np.transpose(returnMembershipArray(xN, centroids))
    sum = 0
    
    for i in range(0, centroids.shape[0]):
        meansPoint = xN[membership[i]==1]
        distance = 0

        for j in range(0, meansPoint.shape[0]):
            distance += euclidean(meansPoint[j], centroids[i])
        
        sum += distance
    return sum

# Classes categorization
def getClasses(xN, centroids):
    membership = np.transpose(returnMembershipArray(xN, centroids))
    classes = np.zeros((3, centroids.shape[0]))

    for i in range(0, membership.shape[0]):
        (values, count) = np.unique(labels[np.where(membership[i]==1)], return_counts=True)
        index = np.argmax(count)
        classes[0][i] = values[index]
        classes[1][i] = count[index]
        classes[2][i] = i
    return classes

# Function to run K-Means on some input array
# given the number of clusters
def kMeansAlgo(xN, clusters):
    oldCentroids = np.zeros((clusters, xN.shape[1]))
    newCentroids = initializeCentroids(xN, clusters)

    # iterating until the algorithm converges.
    while not (np.array_equal(oldCentroids, newCentroids)):
        oldCentroids = newCentroids
        membership = returnMembershipArray(xN, newCentroids)
        newMeans =  updateCentroids(xN, membership, oldCentroids, clusters)
    
    return newMeans

# Corresponding to a main function in Java
def solution(xN, clusters):
    multiMeans = np.zeros((30, clusters, xN.shape[1]))
    minimum = np.inf
    minimumIndex = -1
    
    # Run the algorithm for 30 times,
    # picking the best initialization
    # of centroids
    for i in range(0, multiMeans.shape[0]):
        multiMeans[i] = kMeansAlgo(xN, clusters)
        distance = totalDistance(xN, multiMeans[i])

        if distance < minimum:
            minimum = distance
            minimumIndex = i

    bestMean = multiMeans[minimumIndex]
    classes = getClasses(xN, bestMean)
    return classes

xN , labels = loadData('Images')
clusters = solution(xN, 10)
print(clusters)

# Using pyplot to plot a figure
# representing the counts and values
# of each digit throughout the images
plt.bar(clusters[2], clusters[1], color='blue')
plt.title('K-Means classifier')
plt.xlabel('Digits')
plt.ylabel('Count')
plt.savefig('Counts.jpg')
