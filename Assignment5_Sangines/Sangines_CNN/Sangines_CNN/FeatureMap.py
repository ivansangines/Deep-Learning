import numpy as np
import enumerations

class FeatureMap(object):
    """description of class"""
    def __init__(self, inputDataSize, poolingType, activationFuncType, batchSize ):
        self.inputDataSize = inputDataSize
        self.poolingType = poolingType
        self.activationFuncType = activationFuncType
        self.batchSize = batchSize
        self.poolingSize = inputDataSize//2

        #---------All of the following will be List of np arrays---------
        self.deltaSS = np.zeros((batchSize, self.poolingSize, self.poolingSize)) #Delta at subsampling (pooling)
        self.deltaCV = np.zeros((batchSize, inputDataSize, inputDataSize)) ##Delta of the layer
        self.sumation = np.zeros((batchSize, inputDataSize, inputDataSize)) #result after adding biasses
        self.outputSS = np.zeros((batchSize, self.poolingSize, self.poolingSize)) #Output after subsampling (pooling)
        self.outputSSFinal = np.zeros((batchSize,self.poolingSize*self.poolingSize))
        self.actDeriv = np.zeros((batchSize, inputDataSize, inputDataSize)) #Outputs of Derivative of Activation function
        self.actFuncOut = np.zeros((batchSize, inputDataSize, inputDataSize)) #outputs of Activation Function in current Layer
        #------------------------------------------------------------------

        self.bias = 0.0
        self.biasGrad = 0.0

    def Evaluate(self, inputData, batchIndex):
        self.sumation[batchIndex] = inputData + self.bias #Adding bias to each feature map (sumation is a list of np arrays since inputData will be a numpy array)

        if self.activationFuncType == enumerations.ActivationType.SIGMOID:
            self.actFuncOut[batchIndex] = 1/(1 + np.exp(-self.sumation[batchIndex]))
            self.actDeriv[batchIndex] = self.actFuncOut[batchIndex] * (1 - self.actFuncOut[batchIndex])

        elif self.activationFuncType == enumerations.ActivationType.RELU:
            self.actFuncOut[batchIndex] = np.maximum(0, self.sumation[batchIndex])

        #res = np.zeros((self.poolingSize,self.poolingSize))
        if self.poolingType == enumerations.PoolingType.AVGPOOLING:
            for i in range(self.poolingSize):
                for j in range(self.poolingSize):
                    self.outputSS[batchIndex,i,j] = np.average(self.actFuncOut[batchIndex, i*2:i*2+2, j*2:j*2+2])
            #self.outputSS[batchIndex] = res


        return self.outputSS[batchIndex]

    def Flatten(self): #Apply it to last CNN layer
        for i in range(len(self.outputSS)):
            self.outputSSFinal[i] = self.outputSS[i].flatten() #Flatten last 2 dimensions of OutputSS, i.e. will return a 5x16


        


