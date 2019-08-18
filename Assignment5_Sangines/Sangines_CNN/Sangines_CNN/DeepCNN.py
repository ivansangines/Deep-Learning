import numpy as np

class DeepCNN(object):
    
    def __init__(self, cnnLayerList, layerList, inputDataList, outputLables, batchSize):

        self.cnnLayerList = cnnLayerList
        self.layerList = layerList
        self.inputDataList = inputDataList
        self.outputLables = outputLables
        self.batchSize = batchSize
        self.outputSize = len(cnnLayerList[len(cnnLayerList)-1].featureMapList[0].outputSS[0,0])
        self.flattenSize = self.outputSize * self.outputSize * len(cnnLayerList[len(cnnLayerList)-1].featureMapList)
        self.flatten = np.zeros((batchSize,self.flattenSize))

    def Evaluate(self, inputData, batchIndex):

        self.cnnLayerList[0].Evaluate(inputData, batchIndex) #Doing Kernel convolutions

        for l in range(1,len(self.cnnLayerList )):
            prevOut = [] #List of inputsData

            for fmp in self.cnnLayerList[l-1].featureMapList:
                prevOut.append(fmp.outputSS[batchIndex]) #creating a list of feature maps                
            self.cnnLayerList[l].Evaluate(prevOut, batchIndex)

            if l==len(self.cnnLayerList )-1: #Checking if it is last layer in order to Flatten
                temp = []
                for fm in self.cnnLayerList[l].featureMapList:
                    temp.append(fm.outputSS[batchIndex].flatten())
                self.flatten[batchIndex]  = np.asarray(temp).flatten()
#------------------------------------------------------------------------------------------ FINISH EVALUATE HERE
        '''
        for i in range(len(self.cnnLayerList)):
            prevOut = [] #List of inputsData

            if i==0:
                #prevOut.append(inputData)
                prevOut = inputData
            else:
                prevOut.clear()
                #prevOut = [fmp.outputSS[batchIndex] for fmp in self.cnnLayerList[i-1].featureMapList]
                for fmp in self.cnnLayerList[i-1].featureMapList:
                    prevOut.append(fmp.outputSS[batchIndex]) #creating a list of feature maps
            self.cnnLayerList[i].Evaluate(prevOut, batchIndex) #Doing Kernel convolutions
        

        #--------------------Last Layer Flattering---------------------------
        for bi in range(batchSize):
            temp = []
            for fmp in self.cnnLayerList[i-1].featureMapList:
                temp.append(fmp.outputSS.flatten())
            self.flatten[bi]  = np.asarray(temp).flatten()
        '''
#------------------------------------------------------------------------------------------ CREATE METHOD TO PASS TO NN
        count = 0
        res = None
        for l in self.layerList:
            if count==0:
                res = l.Evaluate(self.flatten[batchIndex], batchIndex, False)
            else:
                res = l.Evaluate(res, batchIndex, False)
            count = count+1
        return res
#-------------------------------------------------------------------------------------------- 
    def Train (self, numEpochs, learningRate, batchSize):

        for i in range(numEpochs): 
            trainingError = 0
            self.inputDataList, self.outputLables = shuffle(self.inputDataList, self.outputLables) #shufle Data

            for j in range(0,len(self.inputDataList), batchSize ): #will go throught all the trainin images, increments by batchSize
            #--------------------IF NOT DOING BATCH, X and Y WILL JUST HAVE 1 IMAGE. ---------------------
            #-----------WHEN DOING BATCHNORM, X AND Y WILL HAVE AS MANY IMAGES AS THE BATCH SIZE----------
                X_train_mini = self.inputDataList[j:j + batchSize]
                y_train_mini = self.outputLables[j:j + batchSize]
                errorM = []*batchSize
                for b in range(batchSize): #Evaluate each img at a time------At the end we will add all the grads etc from each img in the batch
                    res = self.Evaluate(X_train_mini[b], b)
                #PASSS FLATTEN TO NN
                errorM = (res - y_train_mini[b]) * (res - y_train_mini[b])

                


        


