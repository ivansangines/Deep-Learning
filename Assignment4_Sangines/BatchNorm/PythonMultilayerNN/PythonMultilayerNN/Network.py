import numpy as np
import math
from sklearn.utils import shuffle
from Layer import *


class Network(object):
    """description of class"""
    def __init__(self, X, Y, neuronsLayer,batchSize, activaitonF, activationLast ):
        
        self.X = X
        self.Y = Y
        self.activationF = activaitonF
        self.neuronsLayer = neuronsLayer #list with number of neurons per layer (each index represents one layer)
        self.activationLast = activationLast #activation func in last layer
        self.Layers = [] #will cointain all created layers (list of Layers objects)

        #initializing Layers instances
        for i in range(len(self.neuronsLayer)):
            if (i == 0): #First Layer
                layer = (Layer(self.neuronsLayer[i], self.X.shape[1],self.activationF,batchSize,0.8, False,))

            elif (i == (len(self.neuronsLayer)-1)): #Last Layer
                  layer = (Layer(self.Y.shape[1], self.neuronsLayer[i-1],self.activationLast,batchSize,1, True,))

            else: #Mid Layer
                layer =(Layer(self.neuronsLayer[i], self.neuronsLayer[i-1],self.activationF,batchSize,0.8, False,))
            self.Layers.append(layer)

    def Compute(self, input, isBatch = False, isTrain = False):
        
        self.Layers[0].Computations(input, isBatch, isTrain) #computing it separate because the input is X not a from prev layer

        for i in range(1, len(self.neuronsLayer)):
            self.Layers[i].Computations(self.Layers[i-1].a, isBatch, isTrain) #inputs will be the outputs (a) of prev layers

        return self.Layers[len(self.neuronsLayer)-1].a


    def Train (self, epochs, learningRate, gradDesc, optimization="REGULAR", batchSize=1, isBatchNorm = False):

        updates = 0
        for i in range(epochs): 
            error = 0
            self.X, self.Y = shuffle(self.X, self.Y) # shuffle data 

            for j in range(0,self.X.shape[0], batchSize ): #will go throught all the trainin images

                #--------------------IF NOT DOING BATCH, X and Y WILL JUST HAVE 1 IMAGE. ---------------------
                #-----------WHEN DOING BATCHNORM, X AND Y WILL HAVE AS MANY IMAGES AS THE BATCH SIZE----------
                X_train_mini = self.X[j:j + batchSize]
                y_train_mini = self.Y[j:j + batchSize]
                #---------------------------------------------------------------------------------------------
                self.Compute(X_train_mini, isBatchNorm, True) #will call TRAIN in batch

                if (self.activationLast == "SOFTMAX"):
                    error += -0.5*(y_train_mini * np.log(self.Layers[len(self.neuronsLayer)-1].a + 0.001)).sum()
                else:
                    error += 0.5*((self.Layers[len(self.neuronsLayer)-1].a - y_train_mini) * (self.Layers[len(self.neuronsLayer)-1].a - y_train_mini)).sum()

                indexLayer = len(self.neuronsLayer) - 1

                while (indexLayer>=0):
                    if (indexLayer == len(self.neuronsLayer)-1): #last Layer
                        if (self.activationLast == "SOFTMAX"): 
                            self.Layers[indexLayer].delta = -y_train_mini + self.Layers[indexLayer].a
                        else:
                            self.Layers[indexLayer].delta = -(y_train_mini - self.Layers[indexLayer].a) * self.Layers[indexLayer].derivativeAF

                    else: #mid Layer
                        #computing the missing part for delta
                        self.Layers[indexLayer].delta = np.dot(self.Layers[indexLayer+1].delta, self.Layers[indexLayer+1].W) * self.Layers[indexLayer].derivativeAF

                    #---------------------CHECKING IF BATCHNORM-------------------------------------------
                    #will compute derivatives and deltas for BATCHNORM
                    if (isBatchNorm):
                        #COMPUTING DERIVATIVES
                        self.Layers[indexLayer].derivBeta = np.sum(self.Layers[indexLayer].delta)
                        self.Layers[indexLayer].derivGama = np.sum(self.Layers[indexLayer].delta * self.Layers[indexLayer].Shat)
                        self.Layers[indexLayer].deltaBatch = (self.Layers[indexLayer].delta * self.Layers[indexLayer].gamma)/(batchSize * np.sqrt(self.Layers[indexLayer].sigma2+self.Layers[indexLayer].epsilon))*(batchSize - 1 - self.Layers[indexLayer].Shat*self.Layers[indexLayer].Shat)
                   #--------------------------------------------------------------------------------------                        
                        
                    if (indexLayer>0):
                        prevOut = self.Layers[indexLayer-1].a #output from the layer before
                    else:
                        prevOut = X_train_mini #whenever it is first layer, we will use the inputs X

                    if(isBatchNorm):
                        #UPDATES FOR BATCHNORM
                        self.Layers[indexLayer].gradW = np.dot(self.Layers[indexLayer].deltaBatch.T, prevOut)
                        self.Layers[indexLayer].gradB = self.Layers[indexLayer].deltaBatch.sum(axis=0)

                    else:
                        self.Layers[indexLayer].gradW += np.dot(self.Layers[indexLayer].delta, prevOut.T)
                        self.Layers[indexLayer].gradB += self.Layers[indexLayer].delta

                    indexLayer = indexLayer - 1  
                
                #For ADAM Algorithm
                updates = updates + 1
                #Cheching gradient descent type in order to update gradients differently
                if (gradDesc == "MINIBATCH"):
                    if(j%batchSize == 0):
                        self.UpdateGradBias(learningRate, batchSize, updates,isBatchNorm, optimization)
                if (gradDesc == "STOCHASTIC"):
                    self.UpdateGradBias(learningRate, batchSize, updates, isBatchNorm, optimization )

            #For Batch, we will update weights once we finished each epoch. BatchSize is the total number of images
            if (gradDesc == "BATCH"):
                self.UpdateGradBias(learningRate, self.X.shape[0], updates, isBatchNorm, optimization)

            print("Epoch = " + str(i) + "Error =" + str(error))

    def UpdateGradBias (self, learningRate, batchSize, updates, isBatchNorm, optimization="REGULAR"):

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for lay in  range(len(self.neuronsLayer)):
            if (optimization=="REGULAR"):
                res= learningRate * (1/batchSize) * self.Layers[lay].gradW
                self.Layers[lay].W = self.Layers[lay].W - res
                self.Layers[lay].b = self.Layers[lay].b - learningRate * (1/batchSize) * self.Layers[lay].gradB
                self.Layers[lay].ClearGradients()

            elif (optimization=="ADAM"): 
            
                self.Layers[lay].mtw = beta1 * self.Layers[lay].mtw + (1 - beta1) * self.Layers[lay].gradW
                self.Layers[lay].mtb = beta1 * self.Layers[lay].mtb + (1 - beta1) * self.Layers[lay].gradB

                self.Layers[lay].vtw = beta2 * self.Layers[lay].vtw + (1 - beta2) * self.Layers[lay].gradW * self.Layers[lay].gradW
                self.Layers[lay].vtb = beta2 * self.Layers[lay].vtb + (1 - beta2) * self.Layers[lay].gradB * self.Layers[lay].gradB

                mtwFinal = self.Layers[lay].mtw / (1 - beta1**updates )
                mtbFinal = self.Layers[lay].mtb / (1 - beta1**updates )

                vtwFinal = self.Layers[lay].vtw / (1 - beta2**updates )
                vtbFinal = self.Layers[lay].vtb / (1 - beta2**updates )

                self.Layers[lay].W = self.Layers[lay].W - learningRate * (1/batchSize) * mtwFinal/((vtwFinal**0.5) + epsilon)
                self.Layers[lay].b = self.Layers[lay].b - learningRate * (1/batchSize) * mtbFinal/((vtbFinal**0.5) + epsilon)

                self.Layers[lay].ClearGradients()

            if (isBatchNorm == True):
                self.Layers[lay].beta = self.Layers[lay].beta - learningRate *self.Layers[lay].derivBeta
                self.Layers[lay].gamma = self.Layers[lay].gamma - learningRate * self.Layers[lay].derivGama
