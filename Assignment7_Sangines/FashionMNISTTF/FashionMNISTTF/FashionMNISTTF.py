import sys
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def main():
    print(tf.__version__) # tensorflow version
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images.shape # (60000,28,28)
    len(train_labels) # 60000
    test_images.shape # (10000,28,28)
    len(test_labels) # 10000
    plt.figure() # view first training image
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
 
    sys.exit(int(main() or 0))


