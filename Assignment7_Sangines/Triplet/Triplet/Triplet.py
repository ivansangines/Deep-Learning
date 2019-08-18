
from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf 
import matplotlib 
import matplotlib.pyplot as plt 
from Siamese3 import Siamese3 
import numpy as np
from keras.utils import to_categorical 
 
def visualize(embed, labels):     
    labelset = set(labels.tolist())     
    fig = plt.figure(figsize=(8,8))     
    ax = fig.add_subplot(111)     
    #fig, ax = plt.subplots()     
    for label in labelset:         
        indices = np.where(labels == label)         
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)     
    ax.legend()    
    plt.show()     
    plt.close() 
 
def main():     
    # Load MNIST dataset     
    mnist = input_data.read_data_sets('MNIST_data', one_hot = False)     
    mnist_test_labels = mnist.test.labels     
    #mnist_test_onehotlabels = to_categorical(mnist_test_labels) # for onehot outputs 
 
    siamese = Siamese3()     
    siamese.trainSiamese(mnist,5000,100)  # 5000, 128  produces good results     
    #siamese.saveModel()     
    ##siamese.loadModel() 

    siamese.trainSiameseForClassification(mnist,5000,100)         
    # Test model     
    embed = siamese.test_model(input = mnist.test.images)     
    embed = embed.reshape([-1, 2]) 
 
 
    siamese.computeAccuracy(mnist) 
    visualize(embed, mnist_test_labels) 

if __name__ == '__main__':     
    main() 
