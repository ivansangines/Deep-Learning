import numpy as np
#import mnist

class NN(object):
    def __init__(self, numInputs, numHiddenNeurons, numOutputs):
        self.numInputs = numInputs;
        self.numHiddenNeurons = numHiddenNeurons
        self.numOututs = numOutputs
        self.model = {} # dictionary
        W1 = np.random.randn(numInputs, numHiddenNeurons) / np.sqrt(numInputs)
        b1 = np.zeros((1, numHiddenNeurons))
        W2 = np.random.randn(numHiddenNeurons, numOutputs) / np.sqrt(numHiddenNeurons)
        b2 = np.zeros((1, numOutputs))
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    def train(self,X,y, lr=0.1, epochs =100):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        for i in range(0, epochs):
        # Forward propagation
            s1 = X.dot(W1) + b1
            a1 = self.__sigmoid(s1)
            s2 = a1.dot(W2) + b2
            a2 = self.__sigmoid(s2)

            # Backpropagation
            delta2 = (a2 - y)
            delta1 = delta2.dot(W2.T) * a1(1 - a1) 
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
         
            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %(i, self.calculate_loss(model,X,y)))
        return model

    def __sigmoid(self, s): #-- for private method
        return 1 / (1 + np.exp(-s))
