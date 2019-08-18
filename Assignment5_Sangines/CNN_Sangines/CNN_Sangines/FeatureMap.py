import numpy as np
import enumerations

class FeatureMap(object):
    """description of class"""
    def __init__(self, inputDataSize, poolingType, activationFuncType, batchSize=0):
        self.inputDataSize = inputDataSize
        self.poolingType = poolingType
        self.activationFuncType = activationFuncType
        self.batchSize = batchSize
        #---------All of the following will be List of np arrays---------
        self.deltaSS = [batchSize] #Delta at subsampling (pooling)
        self.deltaCV = [batchSize] ##Delta of the layer
        self.sumation = [batchSize] #result after adding biasses
        self.outputSS = [batchSize] #Output after subsampling (pooling)
        self.actDeriv = [batchSize] #Derivative of Activation function
        self.actFuncOut = [batchSize] #Activation Function in current Layer
        #------------------------------------------------------------------
        self.bias = 0.0
        self.biasGrad = 0.0

    def Evaluate(self, inputData, batchIndex):
        self.sumation[batchSize] = inputData + self.bias #Adding bias to each feature map (sumation is a list of np arrays since inputData will be a numpy array)

        if self.activationFuncType == enumerations.ActivationType.SIGMOID:
            self.actFuncOut[batchIndex] = 1/(1 + np.exp(-self.sumation[batchIndex]))
            self.actDeriv[batchIndex] = self.actFuncOut[batchIndex] * (1 - self.actFuncOut[batchIndex])

        elif self.activationFuncType == enumerations.ActivationType.RELU:
            self.actFuncOut[batchIndex] = np.maximum(0, self.sumation[batchIndex])

        if self.poolingType == enumerations.PoolingType.AVGPOOLING:
            res = np.zeros((len(self.actFuncOut[batchIndex])//2,len(self.actFuncOut[batchIndex,0])//2))
            for i in range(len(self.actFuncOut[batchIndex])//2):
                for j in range(len(self.actFuncOut[batchIndex,0])//2):
                    res[i,j] = np.average(self.actDeriv[batchIndex, i*2:i*2+2, j*2:j*2+2])
            self.outputSS[batchIndex] = res
        return 



        


