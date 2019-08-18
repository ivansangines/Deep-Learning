
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from Siamese import SiameseCustom
import numpy as np
from keras.utils import to_categorical
import os
import cv2

def visualize(embed, labels):
    labelset = set(labels.tolist())
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
    ax.legend()
    plt.show()
    plt.close()

def readInputData():
    train = np.empty((60000,28,28),dtype="float64")
    trainYOH = np.zeros((60000,10)) # one hot
    trainY = np.zeros((60000)) # just digit i.e, 5 or 7 etc..
    test = np.empty((10000,28,28),dtype="float64")
    testYOH = np.zeros((10000,10))
    testY = np.zeros((10000))


    i = 0
    for filename in os.listdir("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/TrainingAll60000"):
        y = int(filename[0])
        trainYOH[i,y] = 1.0
        trainY[i] = y
        train[i] = cv2.imread("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/TrainingAll60000/{0}".format(filename),0)/255.0 # for color, use 1
        i = i + 1
        if i%100 == 0:
            print(i)

    i = 0 # read test data
    for filename in os.listdir("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000"):
        y = int(filename[0])
        testYOH[i,y] = 1.0
        testY[i] = y
        test[i] = cv2.imread("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000/{0}".format(filename),0)/255.0
        i = i + 1   
        if i%100 == 0:
            print(i)

    
    #trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2])
    #testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2])
    #return trainX, trainY, trainYOH, testX, testY, testYOH
    return train, trainY, trainYOH, test, testY, testYOH

def main():
    # Load MNIST dataset
    trainX, trainY, trainYOH, testX, testY, testYOH = readInputData() # file based input data
    a = trainX.shape[0]
    print("Loaded data")
    siamese = SiameseCustom() # for custom input data read from images files
    siamese.trainSiamese(trainX, trainY,20,100)
    embed = siamese.test_model(testX)
    embed = embed.reshape([-1, 2])
    siamese.trainSiameseForClassification(trainX, trainY,25,100)
    visualize(embed, testY)
    siamese.computeAccuracy(testX, testY)

if __name__ == "__main__":
    main()
    