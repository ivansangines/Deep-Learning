#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import gzip
import _pickle as cPickle  # Python 3 uses _pickle and not cPickle

def main():
    #mnist = input_data.read_data_sets('MNIST_data', one_hot = False)
    f = gzip.open('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data

    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
 sys.exit(int(main() or 0))