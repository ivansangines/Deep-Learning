from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from Siamese import Siamese
import numpy as np
import gzip
import sys
import _pickle as cPickle  # Python 3 uses _pickle and not cPickle

def visualize(embed, labels):
    labelset = set(labels.tolist())
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
    ax.legend()
    #fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    # Load MNIST dataset
    #mnist = input_data.read_data_sets('MNIST_data', one_hot = False)
    mnist = gzip.open('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(mnist)
    else:
        data = cPickle.load(mnist, encoding='bytes')
    mnist.close()
    (x_train, y_train), (x_test, y_test) = data

    x_train, x_test = x_train / 255.0, x_test / 255.0
    #mnist_test_labels = mnist.test.labels
    
    siamese = Siamese()
    siamese.trainSiamese(x_train,y_train,600,100)

    # Test model
    embed = siamese.test_model(input = x_test)
    #embed.tofile('embed.txt')
    #embed = np.fromfile('embed.txt', dtype = np.float32)
    embed = embed.reshape([-1, 2])
    visualize(embed, y_test)

if __name__ == '__main__':
    main()

