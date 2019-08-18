import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
import numpy as np
from sklearn.utils import shuffle

class SiameseCustom(object):

    def __init__(self):
        #----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network
        self.tf_inputA = tf.placeholder(tf.float32, [None, 28,28], name = "inputA")
        self.tf_inputB = tf.placeholder(tf.float32, [None, 28,28], name = "inputB")

        #self.tf_inputAClass = tf.placeholder(tf.float32, [None, 784], name = "inputA")
        #self.tf_inputBClass = tf.placeholder(tf.float32, [None, 784], name = "inputB")
        

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name = "Y")
        self.tf_YOneHot = tf.placeholder(tf.float32, [None,10], name = "YoneHot")
        # outputs, loss function and training optimizer

        #self.outputA, self.outputB = self.siameseNetwork()
        self.outputA, self.outputFinal = self.siameseNetwork()

        self.output = self.siameseNetworkWithClassification()
        self.loss = self.contastiveLoss()
        self.lossCrossEntropy = self.crossEntropyLoss()
        self.optimizer = self.optimizer_initializer()
        self.optimizerCrossEntropy = self.optimizer_initializer_crossEntropy()
        self.saver = tf.train.Saver()
        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def layerCNN(self,input_data, numChannels, numFeatureMaps, kernelSize, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        # for the first CNN layer, the number of feature maps will be the number of color channles
        # 1 for gray scale, 3 for RGB
        # for the second CNN layer, numChannels will be the number of feature maps in previous CNN layer
        filterShape = [kernelSize[0], kernelSize[1], numChannels, numFeatureMaps] #numChaneels = 3 for RGB

        # initialize weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.03), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([numFeatureMaps]), name=name+'_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        out_layer += bias

        # apply activation function
        output_layer = tf.nn.relu(out_layer)

        # max pooling on output of feature maps
        # ksize = size of the max pooling window (i.e. the area over which the max pooling is done
        # It needs to be 4D to match the data tensor - 1x2x2x1
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        # strides defines how the max pooling area is applied on the image - It also matches the data tensor
        # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
        # strides of 2 in the x and y directions.
        strides = [1, 2, 2, 1]
        output_layer_pooling = tf.nn.max_pool(output_layer, ksize=ksize, strides=strides, padding='SAME')
        return output_layer_pooling

    def layer(self, tf_input, num_hidden_units, variable_name, trainable=True):
        # tf_input: batch_size x n_features
        # num_hidden_units: number of hidden units
        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)
        num_features = tf_input.get_shape()[1]
        W = tf.get_variable(
            name = variable_name + "_W",
            dtype = tf.float32,
            shape = [num_features, num_hidden_units],
            initializer = tf_weight_initializer,
            trainable=trainable
            )
        b = tf.get_variable(
            name = variable_name + "_b",
            dtype = tf.float32,
            shape = [num_hidden_units],
            trainable=trainable
            )
        out = tf.add(tf.matmul(tf_input, W), b)
        return out

    def network(self, tf_input, trainable=True):
        # Setup FNN
        #fc1 = self.layer(tf_input = tf_input, num_hidden_units = 512,trainable=trainable, variable_name = "fc1")
        #ac1 = tf.nn.relu(fc1)
        #fc2 = self.layer(tf_input = ac1, num_hidden_units = 512, trainable=trainable,variable_name = "fc2")
        #ac2 = tf.nn.relu(fc2)

        # reshape the input to a 4D tensor.  The first value = batch size, last = number of channels, 3 for color
        x4D_input = tf.reshape(tf_input, [-1, 28, 28, 1])
    
        # create cnn layers
        numFeatureMapsLayer1 = 8
        numFeatureMapsLayer2 = 16
        cnnLayer1Output = self.layerCNN(x4D_input, 1, numFeatureMapsLayer1, [5, 5], [2, 2], name='cnnLayer1')   
        cnnLayer2Output = self.layerCNN(cnnLayer1Output, numFeatureMapsLayer1, numFeatureMapsLayer2, [5, 5], [2, 2], name='cnnLayer2') 

        flattened = tf.reshape(cnnLayer2Output, [-1, 7 * 7 * numFeatureMapsLayer2]) # first dim is batch size
        # after maxpooling through two layers, the output will be 7x7 x num featuremaps in second CNN layer

        # Create NN layer for embeding
        fc3 = self.layer(tf_input = flattened, num_hidden_units = 100, trainable=trainable,variable_name = "fc3")

        return fc3

    def networkWithClassification(self, tf_input):
        # Setup FNN
        fc3 = self.network(tf_input, trainable=False)
        ac3 = tf.nn.relu(fc3)
        fc4 = self.layer(tf_input = ac3, num_hidden_units = 80, trainable=True,variable_name = "fc4")
        ac4 = tf.nn.relu(fc4)
        fc5 = self.layer(tf_input = ac4, num_hidden_units = 10, trainable=True,variable_name = "fc5")
        return fc5

    def siameseNetwork(self):
        # Initialze neural network
        with tf.variable_scope("siamese") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
            #For new loss function
            outputFin = tf.reduce_sum(tf.abs(tf.subtract(outputA,outputB)))
        return outputA, outputFin
        #return outputA, outputB

    def siameseNetworkWithClassification(self):
        # Initialze neural network
        with tf.variable_scope("siamese",reuse=tf.AUTO_REUSE) as scope:
            #with tf.variable_scope(&quot;siamese&quot;) as scope:
            output = self.networkWithClassification(self.tf_inputA)
        return output

    def contastiveLoss(self, margin = 5.0):
        with tf.variable_scope("siamese") as scope:
            labels = self.tf_Y
            #---------Using new loss function--------------------------
            labels = self.tf_Y 
            p = tf.add(tf.sigmoid(self.outputFinal), 1e-6)
            l = tf.log(tf.add(tf.subtract(1.0,p),1e-6))
            lossSimilar = tf.multiply(tf.subtract(1.0,labels),l)
            lossDissimilar = tf.multiply(labels, tf.log(p))
            loss = tf.reduce_mean(tf.add(lossSimilar,lossDissimilar))
            #-----------------------------------------------------------

            # Euclidean distance squared
            #dist = tf.pow(tf.subtract(self.outputA, self.outputB), 2, name = "Dw")
            #Dw = tf.reduce_sum(dist, 1)
            # add a small value 1e-6 to increase the stability of calculating the gradients for sqrt
            #Dw2 = tf.sqrt(Dw + 1e-6, name = "Dw2")
            # Loss function
            #lossSimilar = tf.multiply(labels, tf.pow(Dw2,2), name = "constrastiveLoss_1")
            #lossDissimilar = tf.multiply(tf.subtract(1.0, labels),
            #tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2), name = "constrastiveLoss_2")
            #loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name ="constrastiveLoss")
        return loss

    def crossEntropyLoss(self):
        labels = self.tf_YOneHot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=labels))
        return lossd

    def optimizer_initializer(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def optimizer_initializer_crossEntropy(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.lossCrossEntropy)
        return optimizer

    def trainSiamese(self, trainX, trainY, numIterations, batchSize=100):
        # Train the network for embeddings via contrastive loss
        for i in range(numIterations):
            trainX, trainY = shuffle(trainX, trainY, random_state=0)
            for j in range(0, trainX.shape[0], batchSize*2):# get (X, y) for current minibatch/chunk
                input1 = trainX[j:j + batchSize]
                y1 = trainY[j:j + batchSize]
                input2 = trainX[j+batchSize:j + batchSize+batchSize]
                y2 = trainY[j+batchSize:j + batchSize+batchSize]
                label = (y1 == y2).astype("float")
                _, trainingLoss = self.sess.run([self.optimizer, self.loss], 
                                                feed_dict = {self.tf_inputA: input1, self.tf_inputB: input2, self.tf_Y:label})
                if i % 1 == 0:
                    print("iteration %d: train loss %.3f" % (i, trainingLoss))
        return trainingLoss
    
    def trainSiameseForClassification(self, trainX, trainY,numIterations, batchSize=10):
        # Train the network for classification via softmax
        for i in range(numIterations):
            trainX, trainY = shuffle(trainX, trainY, random_state=0)
            for j in range(0, trainX.shape[0], batchSize):# get (X, y) for current minibatch/chunk
                input1 = trainX[j:j + batchSize]
                y1 = trainY[j:j + batchSize]
                y1c = to_categorical(y1) # convert labels to one hot
                labels = np.zeros(batchSize)
                _, trainingLoss = self.sess.run([self.optimizerCrossEntropy,self.lossCrossEntropy],
                                                feed_dict = {self.tf_inputA: input1, self.tf_inputB: input1,
                                                             self.tf_YOneHot: y1c, self.tf_Y:labels})
                if i % 5 == 0:
                    print("iteration %d: train loss %.3f" % (i, trainingLoss))
        return trainingLoss

    def test_model(self, input):
        # Test the trained model
        output = self.sess.run(self.outputA, feed_dict = {self.tf_inputA: input})
        return output

    def computeAccuracy(self,testX, testY):
        labels = np.zeros(100)
        yonehot = np.zeros((100,10))
        aout = self.sess.run(self.output, feed_dict={self.tf_inputA: testX,self.tf_inputB: testX,
                                                     self.tf_YOneHot: yonehot, self.tf_Y:labels})
        accuracyCount = 0
        testY = to_categorical(testY) # one hot labels
        for i in range(testY.shape[0]):
            # determine index of maximum output value
            maxindex = aout[i].argmax(axis = 0)
            if (testY[i,maxindex] == 1):
                accuracyCount = accuracyCount + 1
            print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + "%")