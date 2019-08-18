import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
import numpy as np
from sklearn.utils import shuffle

class SiameseCustom(object):

    def __init__(self):
        #--------------------------CNN VARIABLES--------------------------------------------------------------------------
        #----set up place holders for inputs and expected outputs (labels) for the CNN network---
        self.tf_input = tf.placeholder(tf.float32, [None, 28,28], name='X')
        self.tf_y = tf.placeholder(tf.float32, [None, 10], name='y')  # one hot encoding

        # output, loss function and training optimizer
        self.outputBeforeSoftMax, self.output = self.fullCNNNetwork(self.tf_input)  # a2 = output before softmax
        self.loss = self.crossEntropyLoss()
        self.optimizer = self.optimizer_initializer_crossEntropy()
        

        #-----------------------------------------------------------------------------------------------------------------

        #-------------------------SIAMESE VARIABLES-----------------------------------------------------------------------
        # two input placeholders for Siamese network
        self.tf_inputA = tf.placeholder(tf.float32, [None, 784], name = "inputA")
        self.tf_inputB = tf.placeholder(tf.float32, [None, 784], name = "inputB")
        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name = "Y")
        self.tf_YOneHot = tf.placeholder(tf.float32, [None,10], name = "YoneHot")
        # outputs, loss function and training optimizer
        self.outputA, self.outputB = self.siameseNetwork()
        self.output = self.siameseNetworkWithClassification()
        self.lossSiamese = self.contastiveLossSiamese()
        self.lossCrossEntropy = self.crossEntropyLossSiamese()
        self.optimizerSiamese = self.optimizer_initializerSiamese()
        self.optimizerCrossEntropy = self.optimizer_initializer_crossEntropySiamese()
        

        #------------------------------------------------------------------------------------------------------------------
        self.saver = tf.train.Saver()  # for saving trained network

        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forwardPassCNN(self, input):
        x4D_input = tf.reshape(input, [-1, 28, 28, 1])
    
        # create cnn layers
        numFeatureMapsLayer1 = 8
        numFeatureMapsLayer2 = 16
        cnnLayer1Output = self.layerCNN(x4D_input, 1, numFeatureMapsLayer1, [5, 5], [2, 2], name='cnnLayer1')   
        cnnLayer2Output = self.layerCNN(cnnLayer1Output, numFeatureMapsLayer1, numFeatureMapsLayer2, [5, 5], [2, 2], name='cnnLayer2') 

        flattened = tf.reshape(cnnLayer2Output, [7 * 7 * numFeatureMapsLayer2]) # first dim is batch size --------------------------------------- -1,
        return flattened

    def fullCNNNetwork(self, tf_input):
        # reshape the input to a 4D tensor.  The first value = batch size, last = number of channels, 3 for color
        x4D_input = tf.reshape(tf_input, [-1, 28, 28, 1])
    
        # create cnn layers
        numFeatureMapsLayer1 = 8
        numFeatureMapsLayer2 = 16
        cnnLayer1Output = self.layerCNN(x4D_input, 1, numFeatureMapsLayer1, [5, 5], [2, 2], name='cnnLayer1')   
        cnnLayer2Output = self.layerCNN(cnnLayer1Output, numFeatureMapsLayer1, numFeatureMapsLayer2, [5, 5], [2, 2], name='cnnLayer2') 

        flattened = tf.reshape(cnnLayer2Output, [-1, 7 * 7 * numFeatureMapsLayer2]) # first dim is batch size
        # after maxpooling through two layers, the output will be 7x7 x num featuremaps in second CNN layer

        # feed output of flattened second CNNlayer to fully connected layer
        fc1 = self.layer(tf_input = flattened, numNeurons = 50, trainable=True, variableName = 'fc1')
        a1 = tf.nn.relu(fc1)
        fc2 = self.layer(tf_input = a1, numNeurons = 10, trainable=True, variableName = 'fc2')
        a2 = tf.nn.softmax(fc2)
        return fc2, a2  # fc2 is the output before applying the softmax. The optimizer needs this.
                                   
    def crossEntropyLoss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputBeforeSoftMax, labels=self.tf_y))
        return loss

    def optimizer_initializer_crossEntropy(self):
        learningRate = 0.0001
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss)
        return optimizer

    def trainCNN(self, X_train, y_train, X_test, y_test, epochs, batch_size=10):
        total_batch = int(X_train.shape[0] / batch_size)
        for epoch in range(epochs):
            loss = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
            for i in range(total_batch):
                batch_x, batch_y = X_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
                _, trainingLoss = self.sess.run([self.optimizer, self.loss], feed_dict = {self.tf_input: batch_x, self.tf_y: batch_y})
                loss += trainingLoss / total_batch
                #if i % 100 == 0:
                #    print('iteration %d: train loss %.3f' % (i, trainingLoss))
            print("Epoch:", (epoch), "loss =", "{:.3f}".format(loss))
    
        # compute accuracy after training
        # correct_prediction = tf.equal(tf.argmax(self.tf_y, 1), tf.argmax(self.output, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        # accur = self.sess.run(accuracy, feed_dict={self.tf_input: X_test, self.tf_y: y_test})
        # print("Final Accuracy = " + str(accur*100))

    def layer(self, tf_input, numNeurons, variableName, trainable=True):
        # tf_input: batch_size x n_features
        tf_weight_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.03)
        num_features = tf_input.get_shape()[1]
        W = tf.get_variable(
            name = variableName + '_WCNN', 
            dtype = tf.float32, 
            shape = [num_features, numNeurons], 
            initializer = tf_weight_initializer,
            trainable=trainable
            )
        b = tf.get_variable(
            name = variableName + '_bCNN', 
            dtype = tf.float32, 
            shape = [numNeurons],
            trainable=trainable
            )
        out = tf.add(tf.matmul(tf_input, W), b)
        return out  # output of layer without the activation function

    def layerCNN(self,input_data, numChannels, numFeatureMaps, kernelSize, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        # for the first CNN layer, the number of feature maps will be the number of color channles
        # 1 for gray scale, 3 for RGB
        # for the second CNN layer, numChannels will be the number of feature maps in previous CNN layer
        filterShape = [kernelSize[0], kernelSize[1], numChannels, numFeatureMaps] #numChaneels = 3 for RGB

        # initialize weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.03), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([numFeatureMaps]), name=name+'_b')

        input_data = tf.cast(input_data, tf.float32)
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

    #-----------------------------------------------------------------------------------------------------------------------------------

    def layerNN(self, tf_input, num_hidden_units, variable_name, trainable=True):
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
        fc1 = self.layerNN(tf_input = tf_input, num_hidden_units = 512,trainable=trainable, variable_name = "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.layerNN(tf_input = ac1, num_hidden_units = 512, trainable=trainable,variable_name = "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.layerNN(tf_input = ac2, num_hidden_units = 100, trainable=trainable,variable_name = "fc3")
        return fc3

    def networkWithClassification(self, tf_input):
        # Setup FNN
        fc3 = self.network(tf_input, trainable=False)
        ac3 = tf.nn.relu(fc3)
        fc4 = self.layerNN(tf_input = ac3, num_hidden_units = 80, trainable=True,variable_name = "fc4")
        ac4 = tf.nn.relu(fc4)
        fc5 = self.layerNN(tf_input = ac4, num_hidden_units = 10, trainable=True,variable_name = "fc5")
        return fc5

    def siameseNetwork(self):
        # Initialze neural network
        with tf.variable_scope("siamese") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
        return outputA, outputB

    def siameseNetworkWithClassification(self):
        # Initialze neural network
        with tf.variable_scope("siamese",reuse=tf.AUTO_REUSE) as scope:
            #with tf.variable_scope(&quot;siamese&quot;) as scope:
            output = self.networkWithClassification(self.tf_inputA)
        return output

    def contastiveLossSiamese(self, margin = 5.0):
        with tf.variable_scope("siamese") as scope:
            labels = self.tf_Y
            # Euclidean distance squared
            dist = tf.pow(tf.subtract(self.outputA, self.outputB), 2, name = 'Dw')
            Dw = tf.reduce_sum(dist, 1)
            # add 1e-6 to increase the stability of calculating the gradients
            Dw2 = tf.sqrt(Dw + 1e-6, name = 'Dw2')
            # Loss function
            lossSimilar = tf.multiply(labels, tf.pow(Dw2,2), name ='constrastiveLoss_1')
            lossDissimilar = tf.multiply(tf.subtract(1.0, labels),
            tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2), name = 'constrastiveLoss_2')
            loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name ='constrastiveLoss')
        return loss

    def crossEntropyLossSiamese(self):
        labels = self.tf_YOneHot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=labels))
        return lossd

    def optimizer_initializerSiamese(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def optimizer_initializer_crossEntropySiamese(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.lossCrossEntropy)
        return optimizer

    def trainSiamese(self, img_train, lable_train, numIterations, batchSize=100):
        # Train the network
        flat1 = []
        flat2 = []

        #--------------------Obtain Features--------------------------
        #for i in range(img_train.shape[0]):
            #flat1.append(self.forwardPassCNN(img_train[i]))
            #if i%100 == 0:
                #print(i)
            #flat2.append(self.forwardPassCNN(input2[i]))
        #-------------------------------------------------------------

        for j in range(9):
            for i in range(numIterations):
                #input1= np.asarray(img_train[:batchSize]).reshape(batbatchSize,784)
                #input1= np.asarray(img_train[i*batchSize:i*batchSize+batchSize]).reshape(batchSize,784)
                input1= img_train[i*batchSize:i*batchSize+batchSize]
                y1= lable_train[i*batchSize:i*batchSize+batchSize]
                if i == 0:
                    l= img_train[-i*batchSize-batchSize:-1]
                    input2= img_train[-i*batchSize-batchSize-1:-1]
                    y2= lable_train[-i*batchSize-batchSize-1:-1]
                else:
                    input2= img_train[-i*batchSize-batchSize:-i*batchSize]
                    y2= lable_train[-i*batchSize-batchSize:-i*batchSize]
                label = (y1 == y2).astype('float')

                
                #-----------Obtaining features of chosen inputs-----------------------------
                for i in range(input1.shape[0]):
                    flat1.append(self.forwardPassCNN(input1[i]))
                    flat2.append(self.forwardPassCNN(input2[i]))
                #---------------------------------------------------------------------------
                #flat11 = np.asarray(flat1)
                #flat22 = np.asarray(flat2)
                _, trainingLoss = self.sess.run([self.optimizerSiamese, self.lossSiamese], 
                                                feed_dict = {self.tf_inputA: flat1, self.tf_inputB: flat2, self.tf_Y: label})
                if i % 50 == 0:
                    print('iteration %d: train loss %.3f' % (i+500*j, trainingLoss))
        return true
 
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
