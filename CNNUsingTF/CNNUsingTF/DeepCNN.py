import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class DeepCNN(object):

    def __init__(self):
        #----set up place holders for inputs and expected outputs (labels) for the CNN network---
        self.tf_input = tf.placeholder(tf.float32, [None, 28,28], name='X')
        self.tf_y = tf.placeholder(tf.float32, [None, 10], name='y')  # one hot encoding

        # output, loss function and training optimizer
        self.outputBeforeSoftMax, self.output = self.fullCNNNetwork(self.tf_input)  # a2 = output before softmax
        self.loss = self.crossEntropyLoss()
        self.optimizer = self.optimizer_initializer_crossEntropy()

        self.saver = tf.train.Saver()  # for saving trained network
        
        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

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
                _, trainingLoss = self.sess.run([self.optimizer, self.loss], 
                    feed_dict = {self.tf_input: batch_x, self.tf_y: batch_y})
                loss += trainingLoss / total_batch
                #if i % 100 == 0:
                #    print('iteration %d: train loss %.3f' % (i, trainingLoss))
            print("Epoch:", (epoch), "loss =", "{:.3f}".format(loss))
    
        # compute accuracy after training
        correct_prediction = tf.equal(tf.argmax(self.tf_y, 1), tf.argmax(self.output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        accur = self.sess.run(accuracy, feed_dict={self.tf_input: X_test, self.tf_y: y_test})
        print("Final Accuracy = " + str(accur*100))

    def layer(self, tf_input, numNeurons, variableName, trainable=True):
        # tf_input: batch_size x n_features
        tf_weight_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.03)
        num_features = tf_input.get_shape()[1]
        W = tf.get_variable(
            name = variableName + '_W', 
            dtype = tf.float32, 
            shape = [num_features, numNeurons], 
            initializer = tf_weight_initializer,
            trainable=trainable
            )
        b = tf.get_variable(
            name = variableName + '_b', 
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



