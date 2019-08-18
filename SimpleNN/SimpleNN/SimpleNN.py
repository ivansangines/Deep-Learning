import sys
from NN import NN
import gzip
import _pickle as cPickle  # Python 3 uses _pickle and not cPickle


def main():
  
    # download mnsit pkl from https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
    f = gzip.open('C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data
    sh = x_train.shape
    x_train = x_train.reshape(sh[0],sh[1]*sh[2])
    nn = NN(784,50,10)
    nn.train(x_train,y_train)
    a = 5
if __name__ == "__main__":
    sys.exit(int(main() or 0))
