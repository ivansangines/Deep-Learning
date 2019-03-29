import os
import sys
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
    
train = np.empty((1000,28,28),dtype='float64')
trainY = np.zeros((1000,10,1))
test = np.empty((10000,28,28),dtype='float64')
testY = np.zeros((10000,10,1))

#test = np.array([1,2,3,-2,-4,6])
#test[test<0] = 0
#test[test>0]=1
#print(test)

# Load in the images for training
i = 0
for filename in os.listdir('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Training1000/'):
    y = int(filename[0])
    trainY[i,y] = 1.0
    train[i] = cv2.imread('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Training1000/{0}'.format(filename),0)/255.0 # for color, use 1
    i = i + 1

# Creating testing data
i = 0 
for filename in os.listdir('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000'):
    y = int(filename[0])
    testY[i,y] = 1.0
    test[i] = cv2.imread('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment2_Sangines/Data/Test10000/{0}'.format(filename),0)/255.0
    i = i + 1

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2],1)
testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1)

numNeuronsLayer1 = 100
numNeuronsLayer2 = 10
numEpochs = 100
loss_arr = np.ndarray((numEpochs,1))
x_arr = np.ndarray((numEpochs,1))

#-------Randomly initializing weights and bias for each layer of  neurons values between -0.1 and 0.1---------
w1 = np.random.uniform(low=-0.1,high=0.1,size=(numNeuronsLayer1,784))
b1 = np.random.uniform(low=-1,high=1,size=(numNeuronsLayer1,1))
w2 = np.random.uniform(low=-0.1,high=0.1,size=(numNeuronsLayer2,numNeuronsLayer1))
b2 = np.random.uniform(low=-0.1,high=0.1,size=(numNeuronsLayer2,1))
learningRate = 0.005;

gradw2 =0
gradb2 =0
gradw1 =0
gradb1 =0



#-------Training Neurons----------------
for n in range(0,numEpochs): #we will iterate 100 times through all images
    loss = 0
    trainX,trainY = shuffle(trainX, trainY) # shuffle data for stochastic behavior

    for i in range(trainX.shape[0]): #each iteration is an image

        
        # do forward pass
        s1 = np.dot(w1,trainX[i])+b1
        a1 =  np.maximum(0,s1)
        s2 = np.dot(w2,a1)+b2
        a2 =  np.maximum(0,s2)
        # your equations for the forward pass

        # do backprop and compute the gradients * also works instead
        # np.multiply
        #y = list(trainY[i]).index(1)

        loss += (0.5 * ((a2-trainY[i])*(a2-trainY[i]))).sum()
        #loss += (0.5 * np.multiply((a2-trainY[i]),(a2-trainY[i]))).sum()

        # your equations for computing the deltas and the gradients
        relu2_dev = a2.copy()
        relu2_dev[relu2_dev<0]=0
        relu2_dev[relu2_dev>0]=1
        delta2 = -np.multiply(trainY[i]-a2,relu2_dev)

        relu1_dev = a1.copy()
        relu1_dev[relu1_dev<0]=0
        relu1_dev[relu1_dev>0]=1
        delta1 = np.multiply(np.dot(np.transpose(w2),delta2),relu1_dev)

        gradw2 += np.dot(delta2,np.transpose(a1))
        gradb2 += delta2 
        gradw1 += np.dot(delta1, np.transpose(trainX[i]))
        gradb1 += delta1 

        if (i%10==0):
            # adjust the weights
            w2 = w2 - learningRate * gradw2/10
            b2 = b2 - learningRate * gradb2/10
            w1 = w1 - learningRate * gradw1/10
            b1 = b1 - learningRate * gradb1/10
            gradw2 =0
            gradb2 =0
            gradw1 =0
            gradb1 =0

    loss_arr[n,0] = loss;
    x_arr[n,0] = n;
    print("epoch = " + str(n) + " loss = " + (str(loss)))
        
area = 2
colors = ['blue']
plt.scatter(x_arr, loss_arr, s=area, c=colors, alpha=0.5, linewidths=8) #drawing points using X,Y data arrays
plt.title('Linear Least Squares Regression')
plt.xlabel('Number of epocs')
plt.ylabel('Loss')
line, = plt.plot(x_arr, loss_arr, '--', linewidth=2) #line plot
line.set_color('red')
plt.show()

print("done training , starting testing..")

#-----Testing Given Data----------
accuracyCount = 0
for i in range(testY.shape[0]):
    # do forward pass
    s1 = np.dot(w1,testX[i]) + b1
    a1 = np.maximum(0,s1) # np.exp operates on the array
    s2 = np.dot(w2,a1) + b2
    a2 = np.maximum(0,s2)
    # determine index of maximum output value
    a2index = a2.argmax(axis = 0)
    if (testY[i,a2index] == 1):
        accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/10000.0))



