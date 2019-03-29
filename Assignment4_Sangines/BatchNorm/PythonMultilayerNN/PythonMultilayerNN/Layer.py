import numpy as np
class Layer(object):
    """description of class"""
    def __init__(self, numNeurons, numPrevNeurons,  activationFunc, batchSize=1, dropRate=0.2, isLastLayer=False):

        self.numNeurons = numNeurons
        self.numPrevNeurons = numPrevNeurons
        self.activationFunc = activationFunc
        self.lastLayer = isLastLayer
        self.batchSize = batchSize #used for dimentions

        #Creating matrix needed
        self.W = np.random.uniform(-0.1,0.1,(numNeurons,numPrevNeurons))
        self.b = np.random.uniform(-1,1,(self.numNeurons))
        self.delta = np.zeros((numNeurons)) #will contain derivativeAF * all the remaining derivatives
        self.a = np.zeros((numNeurons))
        self.gradW = np.zeros((numNeurons,numPrevNeurons))
        self.gradB = np.zeros((numNeurons))
        self.derivativeAF = np.zeros((numNeurons)) #will be part of delta, a*deriv of activation func. Used in back propagation
        self.dropRate = dropRate
        self.dropOut = None

        #--------------------ADAM variables-------------------------------------------
        self.mtw = np.zeros((self.numNeurons, self.numPrevNeurons))
        self.mtb = np.zeros((self.numNeurons))
        self.vtw = np.zeros((self.numNeurons, self.numPrevNeurons))
        self.vtb = np.zeros((self.numNeurons))
        #-----------------------------------------------------------------------------

        #-------------------BATCH NORMALIZATION---------------------------------------
        self.mu = np.zeros((self.numNeurons))
        self.sigma2 = np.zeros((self.numNeurons))
        self.epsilon = 1e-6
        self.gamma = np.random.rand(1)
        self.beta = np.random.rand(1)
        self.S = np.zeros((self.numNeurons))
        self.Shat = np.zeros((self.numNeurons))
        self.Sb = np.zeros((self.numNeurons))
        self.muRun = np.zeros((self.numNeurons))
        self.sigma2Run = np.zeros((self.numNeurons))
        self.derivGama = np.zeros((self.numNeurons))
        self.derivBeta = np.zeros((self.numNeurons))
        self.deltaBatch = np.zeros((self.numNeurons))        
        #-----------------------------------------------------------------------------

        
    def Computations(self, input, isBatch = False, isTrain = False):
        self.S = np.dot(input, self.W.T) + self.b

        if (isBatch):
            #We have to diferenciate between Train and Test
            #-----------------------TRAIN------------------------------------
            if(isTrain):
                self.mu = np.mean(self.S)
                self.sigma2 = np.var(self.S)
                self.muRun = 0.9 * self.muRun + (1 - 0.9)* self.mu
                self.sigma2Run = 0.9 * self.sigma2Run + (1 - 0.9)* self.sigma2
            #------------------TEST-------------------------------------------
            else:
                self.mu = self.muRun
                self.sigma2 = self.sigma2Run
            self.Shat =  (self.S - self.mu)/np.sqrt(self.sigma2 + self.epsilon)
            self.Sb = self.Shat * self.gamma + self.beta
            sum = self.Sb
        else:
            sum = self.S
        

        if (self.activationFunc == "SIGMOID"):
            self.a = 1/(1 + np.exp(-sum)) #applying Sigmoid func to each Sum for each neuron
            self.derivativeAF = self.a * (1 - self.a) #used in back propagation, part of the delta

        elif (self.activationFunc == "TANH"):
            self.a = np.tanh(sum)
            self.derivativeAF = (1-self.a*self.a)

        elif (self.activationFunc == "RELU"):
            self.a = np.maximum(0,sum)
            #self.derivativeAF = 1.0 * (self.a>0) #1 * TRUE=1 and 1*FALSE = 0
            epsilon = 1.06e-6
            self.derivativeAF = 1. * (self.a > epsilon)
            self.derivativeAF[self.derivativeAF == 0] = epsilon

        elif (self.activationFunc == "SOFTMAX"):
            
            if (sum.shape[0] == sum.size):
                ex = np.exp(sum)
                self.a = ex/ex.sum()
                
            else:
                ex = np.exp(sum)
                for i in range (ex.shape[0]):
                    denom = ex[i,:].sum()
                    ex[i,:] = ex[i,:]/denom
                    self.a = ex            
            self.derivativeAF = None

        #DropOut
        if (self.lastLayer == False):
            self.dropOut = np.random.binomial(1,self.dropRate,(self.numNeurons))/self.dropRate
            self.a = self.a*self.dropOut 
            self.derivativeAF = self.dropOut * self.derivativeAF


    def ClearGradients(self): #In order to set gradients to 0 after each computed Image
        self.gradB = np.zeros((self.numNeurons))
        self.gradW = np.zeros((self.numNeurons,self.numPrevNeurons))

        


