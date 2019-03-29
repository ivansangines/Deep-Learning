import numpy as np
class Layer(object):
    """description of class"""
    def __init__(self, numNeurons, numPrevNeurons,  activationFunc, dropRate=0.2, isLastLayer=False):

        self.numNeurons = numNeurons
        self.numPrevNeurons = numPrevNeurons
        self.activationFunc = activationFunc
        self.lastLayer = isLastLayer

        #Creating matrix needed
        self.W = np.random.uniform(-0.1,0.1,(numNeurons,numPrevNeurons))
        self.b = np.random.uniform(-1,1,(numNeurons,1))
        self.delta = np.zeros((numNeurons,1)) #will contain derivativeAF * all the remaining derivatives
        self.a = np.zeros((numNeurons,1))
        self.gradW = np.zeros((numNeurons,numPrevNeurons))
        self.gradB = np.zeros((numNeurons,1))
        self.derivativeAF = np.zeros((numNeurons,1)) #will be part of delta, a*deriv of activation func. Used in back propagation
        self.dropRate = dropRate
        self.dropOut = None

        #--------------------ADAM variables-------------------------------------------
        self.mtw = np.zeros((self.numNeurons, self.numPrevNeurons))
        self.mtb = np.zeros((self.numNeurons,1))
        self.vtw = np.zeros((self.numNeurons, self.numPrevNeurons))
        self.vtb = np.zeros((self.numNeurons,1))
        #-----------------------------------------------------------------------------

        
    def Computations(self, input):
        sum = np.dot(self.W,input) + self.b
        

        if (self.activationFunc == "SIGMOID"):
            self.a = 1/(1 + np.exp(-sum)) #applying Sigmoid func to each Sum for each neuron
            self.derivativeAF = self.a * (1 - self.a) #used in back propagation, part of the delta

        elif (self.activationFunc == "TANH"):
            self.a = np.tanh(sum)
            self.derivativeAF = (1-self.a*self.a)

        elif (self.activationFunc == "RELU"):
            self.a = np.maximum(0,sum)
            self.derivativeAF = 1.0 * (self.a>0) #1 * TRUE=1 and 1*FALSE = 0

        elif (self.activationFunc == "SOFTMAX"):
            exp = np.exp(sum)
            self.a = exp/exp.sum()
            self.derivativeAF = None
        #DropOut
        if (self.lastLayer == False):
            self.dropOut = np.random.binomial(1,self.dropRate,(self.numNeurons,1))/self.dropRate
            self.a = self.dropOut * self.a
            self.derivativeAF = self.dropOut * self.derivativeAF


    def ClearGradients(self): #In order to set gradients to 0 after each computed Image
        self.gradB = np.zeros((self.numNeurons,1))
        self.gradW = np.zeros((self.numNeurons,self.numPrevNeurons))

        


