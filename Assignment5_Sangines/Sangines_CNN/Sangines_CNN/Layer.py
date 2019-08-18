import numpy as np
import enumerations

#LAYER FOR A REGULAR NN
class Layer(object):
    def __init__(self, numNeurons, inputSize, activationType, batchSize, dropout=1.0, momentumBN=0.8):
        
        self.numNeurons = numNeurons#
        self.batchSize = batchSize#
        self.dropOut = dropout#
        self.momentumBN = momentumBN#
        self.activationType = activationType#

        self.W = np.random.uniform(-0.1,0.1,(numNeurons,inputSize))#
        self.b = np.random.uniform(-1,1,(numNeurons,1))#
        self.delta = np.zeros((batchSize,numNeurons))#
        self.a = np.zeros((batchSize,numNeurons))#
        self.gradW = np.zeros((batchSize,numNeurons,inputSize))#
        self.gradB = np.zeros((batchSize,numNeurons,1))#
        self.derivAF = np.zeros((batchSize, numNeurons))#
        self.Sum = np.zeros((batchSize, numNeurons))#
        self.dropOutM = None#

        #----------------BATCH NORMALIZATION----------------
        self.mu = np.zeros((numNeurons))#
        self.sigma2 = np.zeros((numNeurons))#
        self.gamma = np.random.rand(1)#
        self.beta = np.random.rand(1)#
        self.epsilon = 1e-8#
        self.sigma2Run = np.zeros((numNeurons))#
        self.muRun = np.zeros((numNeurons))#
        self.derivGama = np.zeros((numNeurons))#
        self.derivBeta = np.zeros((numNeurons))#
        self.Shat = np.zeros((batchSize, numNeurons))#
        self.Sb = np.zeros((batchSize, numNeurons))
        #-----------------------------------------------------

        def Evaluate(self, inputData, batchInd, useBatchNorm=True, isTest=False):
            self.Sum[batchInd] = np.dot(inputData, self.W.T) + self.b

            if (isBatch):
            #We have to diferenciate between Train and Test
            #-----------------------TRAIN------------------------------------
                if(isTrain):
                    self.mu = np.mean(self.Sum[batchInd])
                    self.sigma2 = np.var(self.Sum[batchInd])
                    self.muRun = 0.9 * self.muRun + (1 - 0.9)* self.mu
                    self.sigma2Run = 0.9 * self.sigma2Run + (1 - 0.9)* self.sigma2
            #------------------TEST-------------------------------------------
                else:
                    self.mu = self.muRun
                    self.sigma2 = self.sigma2Run
            #------------------------------------------------------------------
                self.Shat[batchInd] =  (self.Sum[batchInd] - self.mu)/np.sqrt(self.sigma2 + self.epsilon)
                self.Sb[batchInd] = self.Shat[batchInd] * self.gamma + self.beta
                sum = self.Sb[batchInd]
            else:
                sum = self.Sum[batchInd]

            if (self.activationType == enumerations.ActivationType.SIGMOID):
                self.a[batchInd] = 1/(1 + np.exp(-sum)) #applying Sigmoid func to each Sum for each neuron
                self.derivAF[batchInd] = self.a[batchInd] * (1 - self.a[batchInd]) #used in back propagation, part of the delta
            elif (self.activationType == enumerations.ActivationType.RELU):
                self.a[batchInd] = np.maximum(0,sum)
                epsilon = 1.06e-6
                self.derivAF[batchInd] = 1. * (self.a[batchInd] > epsilon)
                temp = self.derivAF[batchInd]
                temp[self.derivAF[batchInd] == 0] = epsilon
                self.derivAF[batchInd] = temp

            elif (self.activationType == enumerations.ActivationType.SOFTMAX):
            
                if (sum.shape[0] == sum.size):
                    ex = np.exp(sum)
                    self.a[batchInd] = ex/ex.sum()
                else:
                    ex = np.exp(sum)
                    for i in range (ex.shape[0]):
                        denom = ex[i,:].sum()
                        ex[i,:] = ex[i,:]/denom
                        self.a[batchInd] = ex            
                self.derivAF[batchInd] = None
               
            #DropOut
            if (self.dropOut < 1.0):
                self.dropOutM = np.zeros((batchSize, numNeurons))
                self.dropOutM = [np.random.binomial(1,self.dropRate,(self.numNeurons))/self.dropRate for i in range(len(self.dropOutM))]
                self.a[batchInd] = self.a[batchInd]*self.dropOutM[batchInd]
                self.derivativeAF[batchInd] = self.dropOutM[batchInd] * self.derivativeAF[batchInd]
            return self.a[batchInd]
            