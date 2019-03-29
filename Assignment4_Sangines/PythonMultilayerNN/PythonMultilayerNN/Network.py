import numpy as np
import math
from sklearn.utils import shuffle
from Layer import *


class Network(object):
    """description of class"""
    def __init__(self, X, Y, neuronsLayer, activaitonF, activationLast ):
        
        self.X = X
        self.Y = Y
        self.activationF = activaitonF
        self.neuronsLayer = neuronsLayer #list with number of neurons per layer (each index represents one layer)
        self.activationLast = activationLast #activation func in last layer
        self.Layers = [] #will cointain all created layers (list of Layers objects)

        #initializing Layers instances
        for i in range(len(self.neuronsLayer)):
            if (i == 0): #First Layer
                layer = (Layer(self.neuronsLayer[i], self.X.shape[1],self.activationF,0.8, False,))

            elif (i == (len(self.neuronsLayer)-1)): #Last Layer
                  layer = (Layer(self.Y.shape[1], self.neuronsLayer[i-1],self.activationLast,1, True,))

            else: #Mid Layer
                layer =(Layer(self.neuronsLayer[i], self.neuronsLayer[i-1],self.activationF,0.8, False,))
            self.Layers.append(layer)

    def Compute(self, input):
        self.Layers[0].Computations(input) #computing it separate because the input is X not a from prev layer

        for i in range(1, len(self.neuronsLayer)):
            self.Layers[i].Computations(self.Layers[i-1].a) #inputs will be the outputs (a) of prev layers

        return self.Layers[len(self.neuronsLayer)-1].a

    def Train (self, epochs, learningRate, gradDesc, optimization="REGULAR", batchSize=1):

        updates = 0
        for i in range(epochs): 
            error = 0
            self.X, self.Y = shuffle(self.X, self.Y) # shuffle data 

            for j in range(self.X.shape[0]): #will go throught all the trainin images
                self.Compute(self.X[j]) #will create layers and doing forward pass

                if (self.activationLast == "SOFTMAX"):
                    error += -0.5*(self.Y[j] * np.log(self.Layers[len(self.neuronsLayer)-1].a + 0.001)).sum()
                else:
                    error += 0.5*((self.Layers[len(self.neuronsLayer)-1].a - self.Y[j]) * (self.Layers[len(self.neuronsLayer)-1].a - self.Y[j])).sum()

                indexLayer = len(self.neuronsLayer) - 1

                while (indexLayer>=0):
                    if (indexLayer == len(self.neuronsLayer)-1): #last Layer
                        if (self.activationLast == "SOFTMAX"): 
                            self.Layers[indexLayer].delta = -self.Y[j] + self.Layers[indexLayer].a
                        else:
                            self.Layers[indexLayer].delta = -(self.Y[j] - self.Layers[indexLayer].a) * self.Layers[indexLayer].derivativeAF

                    else: #mid Layer
                        #computing the missing part for delta
                        self.Layers[indexLayer].delta = np.dot(self.Layers[indexLayer+1].W.T, self.Layers[indexLayer+1].delta) * self.Layers[indexLayer].derivativeAF

                    if (indexLayer>0):
                        prevOut = self.Layers[indexLayer-1].a #output from the layer before
                    else:
                        prevOut = self.X[j] #whenever it is first layer, we will use the inputs X

                    self.Layers[indexLayer].gradW += np.dot(self.Layers[indexLayer].delta, prevOut.T)
                    self.Layers[indexLayer].gradB += self.Layers[indexLayer].delta

                    indexLayer = indexLayer - 1  
                
                #For ADAM Algorithm
                updates = updates + 1
                #Cheching gradient descent type in order to update gradients differently
                if (gradDesc == "MINIBATCH"):
                    if(j%batchSize == 0):
                        self.UpdateGradBias(learningRate, batchSize, updates, optimization )
                if (gradDesc == "STOCHASTIC"):
                    self.UpdateGradBias(learningRate, batchSize, updates, optimization )

            #For Batch, we will update weights once we finished each epoch. BatchSize is the total number of images
            if (gradDesc == "BATCH"):
                self.UpdateGradBias(learningRate, self.X.shape[0], updates, optimization)

            print("Epoch = " + str(i) + "Error =" + str(error))

    def UpdateGradBias (self, learningRate, batchSize, updates, optimization="REGULAR"):

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        if (optimization=="REGULAR"):
            for layer in  range(len(self.neuronsLayer)):
                self.Layers[layer].W = self.Layers[layer].W - learningRate * (1/batchSize) * self.Layers[layer].gradW
                self.Layers[layer].b = self.Layers[layer].b - learningRate * (1/batchSize) * self.Layers[layer].gradB
                self.Layers[layer].ClearGradients()

        elif (optimization=="ADAM"): 
            
            for lay in  range(len(self.neuronsLayer)):
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





