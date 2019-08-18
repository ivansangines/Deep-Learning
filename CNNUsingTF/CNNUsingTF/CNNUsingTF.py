import gzip
import sys
import _pickle as cPickle
from keras.utils import to_categorical
import tensorflow as tf
from DeepCNN import DeepCNN

def main():
    #mnist = tf.keras.datasets.mnist
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    mnist = gzip.open('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(mnist)
    else:
        data = cPickle.load(mnist, encoding='bytes')
    mnist.close()
    (x_train, y_train), (x_test, y_test) = data

    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    learningRate = 0.0001
    numEpochs = 10
    batchSize = 50
    deepCNN = DeepCNN()
    deepCNN.trainCNN(x_train, y_train, x_test, y_test, 10,50)
    print('done..')

if __name__ == "__main__":
    sys.exit(int(main() or 0))
