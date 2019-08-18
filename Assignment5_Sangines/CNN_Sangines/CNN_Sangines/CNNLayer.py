import numpy as np



class CNNLayer(object):

    def __init__(self, numFeatureMaps, numPrevLayerFeatureMaps, inputSize, kernelSize, poolingType, activationType, batchSize):
        self.batchSize = batchSize #Images passing
        self.convolSum = np.zeros((batchSize,numFeatureMaps)) #Each image (in batch) will have one Sum for each FeatureMap we want. 

        self.kernelSizes = kernelSize
        self.numFeatureMaps = numFeatureMaps #Features Maps we want on the current Layer
        self.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps #Feature Maps coming from previous Layer.

        self.convolResults = np.zeros((batchSize,numPrevLayerFeatureMaps, numFeatureMaps)) #Each img in the Batch will have an array 
                                                                                           #Each array represents an output Feature from pev layer
                                                                                           #Each prev feature will have sizes equal to the Features needed in current Layer
        self.featureMapList = [] #List of feature maps in the actual Layer
        for i in range(self.numFeatureMaps): #Looping though all the feature maps needed for the Layer
            featureMp = FeatureMap(convOutputSize, poolingType, activationType, batchSize) #creating Feature Map
            self.featureMapList.append(featureMp)

        self.convOutputSize = inputSize - kernelSize +1 #Sizes after applying Convolution
        self.kernels = np.zeros((numPrevLayerFeatureMaps, numFeatureMaps, kernelSize, kernelSize)) #Kernels matrix will have sizes of PrevFeatures x CurrentFeatures
        self.kernelsGrads = np.zeros((numPrevLayerFeatureMaps, numFeatureMaps, kernelSize, kernelSize))


        #In each img in batch from ConvolSum, we will store X matrixes of size (convolRes x convolRes) where X = Total Features we want in the Layer
        for i in range(batchSize):
            for j in range(numFeatureMaps):
                self.convolSum[i, j] = np.zeros((self.convOutputSize,self.convOutputSize))
        
        #Initializing Kernels 
        self.kernels = InitializeKernel(numPrevLayerFeatureMaps,numFeatureMaps, kernelSize)
        #self.kernels = InitMatrixesInArray(numPrevLayerFeatureMaps,numFeatureMaps, kernelSize)
        #self.kernelsGrads = InitMatrixesInArray(numPrevLayerFeatureMaps,numFeatureMaps, kernelSize)

    #creating empty matrixes for kernels
    #def InitMatrixesInArray(self, dimPrev, dimActual, subDim):
        #tempKern = [np.zeros((subDim,subDim)) for i in range(dimPrev*dimActual)]
        #return np.asarray(tempKern)

    #Initialize Kernels
    #def InitializeKernels(self):
        #self.kernels = [InitializeKernel(self.kernels[i] for i in range(len(self.kernels)))]
        #for i in range(len(self.kernels)):
        #    InitializeKernel(self.kernels[i])

    def InitializeKernel(self, dim1, dim2, kSize):
        temp = np.zeros((dim1, dim2, kSize, kSize))
        for i in range(len(temp[0])):
            for j in range(len(temp[0,0])):
                for a in range(len(temp[0,0,0])):
                    temp[i,j,a] = [np.random.uniform(0,1) * 0.1 if np.random.uniform(0,1)<0.5 else np.random.uniform(0,1) * -0.1 for item in temp[i,j,a]]

                #num = np.random.uniform(0,1)
                #if num<0.5:
                    #temp[i,j] = np.random.uniform(0,1) * 0.1
                #else:
                    #temp[i,j] = np.random.uniform(0,1) * -0.1
        return temp

    def Evaluate(self, PrevLayerOutput, batchIndex):
        #Doing convolutions with the features (outputs) from the Prev Layer
        for i in range(self.numPrevLayerFeatureMaps): #Each iteration is a PrevFeature Map 
            for j in range(self.numFeatureMaps): #Number of Feature Maps we want in current Layer
                self.convolResults[batchIndex, i, q] = PrevLayerOutput[i].Convolution(kernels[i,j])




            



