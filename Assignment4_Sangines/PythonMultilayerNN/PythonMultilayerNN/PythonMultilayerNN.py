import os
import sys
import cv2
import numpy as np
from sklearn.utils import shuffle
from Network import *

def main():
    train = np.empty((1000,28,28),dtype='float64')
    trainY = np.zeros((1000,10,1))
    test = np.empty((10000,28,28),dtype='float64')
    testY = np.zeros((10000,10,1)) # Load in the images
    i = 0 

    for filename in os.listdir('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Training1000/'):
        y = int(filename[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Training1000/{0}'.format(filename),0)/255.0
        i = i + 1
    i = 0 # read test data
    for filename in os.listdir('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000'):
        y = int(filename[0])
        testY[i,y] = 1.0
        test[i] = cv2.imread('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000/{0}'.format(filename),0)/255.0
        i = i + 1
    trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2],1)
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1) 

    numLayers = [50,10]
    NN = Network(trainX,trainY,numLayers, "RELU", "SOFTMAX") 
    NN.Train(10, 0.002, "STOCHASTIC", "ADAM")
    print("done training , starting testing..") 

    accuracyCount = 0
    for i in range(testY.shape[0]):
        # do forward pass
        a2 = NN.Compute(testX[i])
        # determine index of maximum output value
        maxindex = a2.argmax(axis = 0)
        if (testY[i,maxindex] == 1):
            accuracyCount = accuracyCount + 1
    print("Accuracy count = " + str(accuracyCount/10000.0)) 

if __name__ == "__main__":
 sys.exit(int(main() or 0))
