import numpy as np
import FeatureMap
from scipy import signal


class CNNLayer(object):

    def __init__(self, numFeatureMaps, numPrevLayerFeatureMaps, inputSize, kernelSize, poolingType, activationType, batchSize):
        
        self.batchSize = batchSize #Images passing
        self.kernelSizes = kernelSize
        self.numFeatureMaps = numFeatureMaps #Features Maps we want on the current Layer
        self.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps #Feature Maps coming from previous Layer.

                                                                  #Each array represents an output Feature from pev layer
                                                                  #Each prev feature index will have sizes equal to the Features needed in current Layer
        self.featureMapList = [] #List of feature maps in the actual Layer
        
        self.convOutputSize = inputSize - kernelSize +1 #Sizes after applying Convolution
        self.convolResults = np.zeros((batchSize, numPrevLayerFeatureMaps, numFeatureMaps, self.convOutputSize, self.convOutputSize))  #Each img in the Batch will have an array 
        self.convolSum = np.zeros((batchSize, numFeatureMaps, self.convOutputSize, self.convOutputSize))  #Each image (in batch) will have one Sum for each FeatureMap we want. 

        for i in range(self.numFeatureMaps): #Looping though all the feature maps needed for the Layer
            featureMp = FeatureMap(self.convOutputSize, poolingType, activationType, batchSize) #creating Feature Map
            self.featureMapList.append(featureMp)

        
        self.kernels = np.zeros((numPrevLayerFeatureMaps, numFeatureMaps, kernelSize, kernelSize)) #Kernels matrix will have sizes of PrevFeatures x CurrentFeatures
        self.kernelsGrads = np.zeros((numPrevLayerFeatureMaps, numFeatureMaps, kernelSize, kernelSize))


        
        #Initializing Kernels 
        self.kernels = InitializeKernel(numPrevLayerFeatureMaps,numFeatureMaps, kernelSize)
        

    def InitializeKernel(self, dim1, dim2, kSize):
        #temp = np.zeros((dim1, dim2, kSize, kSize))
        for i in range(len(self.kernels[0])):
            for j in range(len(self.kernels[0,0])):
                for a in range(len(self.kernels[0,0,0])):
                    self.kernels[i,j,a] = [np.random.uniform(0,1) * 0.1 if np.random.uniform(0,1)<0.5 else np.random.uniform(0,1) * -0.1 for item in self.kernels[i,j,a]]

                #num = np.random.uniform(0,1)
                #if num<0.5:
                    #temp[i,j] = np.random.uniform(0,1) * 0.1
                #else:
                    #temp[i,j] = np.random.uniform(0,1) * -0.1
        #return temp

    def Evaluate(self, PrevLayerOutput, batchIndex): #PASSING 5 IMAGES (batch size=5)

        #Doing convolutions with the features (outputs) from the Prev Layer
        for i in range(self.numPrevLayerFeatureMaps): #Each iteration is a PrevFeature Map 
            for j in range(self.numFeatureMaps): #Number of Feature Maps we want in current Layer
                self.convolResults[batchIndex, i, j] = signal.convolve2d( PrevLayerOutput, kernels[i,j], mode="valid") #NEED TO PUT INDEX???????
        
 
        #Adding convolution results (all 1st ones together and so on)
        for i in range(len(self.featureMapList)): #total iterations=numFeatures in current Layer
            for j in range(len(PrevLayerOutput)): #total iterations=numFeatures PrevLayer
                self.convolSum[batchIndex,i] = self.convolSum[batchIndex,i] + self.convolResults[j,i]
        
        #Evaluate each feature map in the current Layer (apply ActvFunc and add Bias)
        for i in range(len(self.featureMapList)):
            self.featureMapList[i].Evaluate(self.convolSum[batchIndex,i], batchIndex)



            



