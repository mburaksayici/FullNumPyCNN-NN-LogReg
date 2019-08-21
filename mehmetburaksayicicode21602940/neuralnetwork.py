import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import random
np.random.seed(42)
from numba import jit,njit


#Data Prep---
x_train, y_train, x_test, y_test = mnist.load()
x_train = x_train.reshape(-1,28*28)


numbofimages = 60000
mnistx = x_train/255
mnisty = y_train.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnisty] = 1
mnisty = b
#---


mnistxtest = x_test/255
numbofimages = 10000
mnistytest = y_test.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnistytest] = 1
mnistytest = b






def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)



def relu(x):
    x.copy()
    out = x*(x > 0)
    return out



def relu_derivative(x):

    out = x.copy()
    out[out <= 0] = 0
    return out


print("oluyo")


def cceforward(x,y,w1,w2,lambdaa = 1e-5,reg = "CCE"):
    if reg == "CCE":
        return np.sum(-y * np.log(x +  1e-15) / x.shape[0]) # x + 1e-15
    elif  reg == "CCEL2":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0])+ lambdaa*np.sum(np.square(w2))# + lambdaa*np.sum(np.square(w1))
    elif reg == "CCEL1":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0]) + lambdaa*np.sum(np.abs(w2))  # +  lambdaa*np.sum(np.abs(w1))





@jit
def train(losstype="CCEL1",lr=0.01, epoch= 10,  batchsize=32, weightdecay = 1,weightdecayperiod= 10,randomweights = True,
          node=700,testdatasetx=mnistx[48000:60000],testdatasety = mnisty[48000:60000]): #testdatasetx=mnistxtest[0:10000],testdatasety = mnistytest[0:10000]):



    if randomweights == True:
        w1 = np.random.normal(0.0, 1 , (784, node))
        w2 = np.random.normal(0.0,  0.07216878364  , (node, 10))

        b1 = np.random.normal(0.0,1, (1, node))
        b2 = np.random.normal(0.0, 0.07216878364, (1, 10))



    traininglosstable = np.array([])
    testinglosstable = np.array([])
    trainingepochtable = np.array([])
    testingepochtable = np.array([])

    for i in range(epoch):
        beforeepoch = time.time()

        if i % weightdecayperiod == 0 and i > 1:
            lr = lr * weightdecay

        trainingepochaccuracy = 0
        testingepochaccuracy = 0

        lambdaa = 1e-5
        howmanyimagesfortraining = 30000 # 60000
        repet = int(howmanyimagesfortraining / batchsize)
        losssum = 0


        for a in range(repet):

            mnistxtrain = mnistx[batchsize * a:(batchsize * (a + 1))]
            mnistytrain = mnisty[batchsize * a:(batchsize * (a + 1))]

            layer1output = np.dot(mnistxtrain, w1) 
            layer1outputrelu = relu(layer1output)
            #print(layer1outputrelu)

            pred = softmax(np.dot(layer1outputrelu, w2) + 0)

            # For computational efficiency
            w2temp = w2.copy()
            # layer1outputsigmoid = sigmoid(layer1output)
            predminusy = (pred-mnistytrain)
            reluder = relu_derivative(predminusy)

            reluderw2 = np.dot(reluder,w2temp.T)

            xreluderw2 = np.dot(mnistxtrain.T,reluderw2)
            #predminusyw2 = np.dot(predminusy , w2temp.T )

            




            #print("A:",a)
            if losstype == "CCE":
                loss = cceforward(pred, mnistytrain,w1,w2, reg=losstype, lambdaa=lambdaa)
                losssum = losssum + loss
                #print(losstype)


                w2 = w2 - lr *(np.dot(layer1outputrelu.T, predminusy))#ok for relu / batchsize  # - lambdaa*w2 #  - lambdaa*abs(w2)/w2   LASSO


                b2 = b2 - lr * ((np.mean(predminusy, axis=0, keepdims=True))  / batchsize )# *0.001  # -lambdaa*abs(b2)/b2                 # ok for relu


                w1 = w1-  lr *(xreluderw2/ batchsize ) # -lambdaa*w1


                b1 = b1 - lr * (np.mean( reluderw2, keepdims=True, axis=0) / batchsize  )#*0.001



            elif losstype == "CCEL2":
                loss = cceforward(pred,mnistytrain, w1,w2,   reg=losstype, lambdaa=lambdaa)
                losssum = losssum + loss
                w2 = w2 - lr *(np.dot(layer1outputrelu.T, predminusy)) - lambdaa*w2 #  - lambdaa*abs(w2)/w2   LASSO

                b2 = b2 - lr * ((np.mean(predminusy, axis=0, keepdims=True))  / batchsize ) # - lambdaa*abs(b2)/b2

                w1 = w1 - lr *(xreluderw2/ batchsize ) #  -lambdaa*w1

                b1 = b1 - lr * (np.mean( reluderw2, keepdims=True, axis=0))

            elif losstype == "CCEL1":

                loss = cceforward(pred, mnistytrain, w1,w2, reg=losstype, lambdaa=lambdaa)
                losssum = losssum + loss

                w2 = w2 - lr *(np.dot(layer1outputrelu.T, predminusy)) - lambdaa*abs(w2)/w2

                b2 = b2 - lr * ((np.mean(predminusy, axis=0, keepdims=True))  / batchsize )

                w1 = w1-  lr *(xreluderw2/ batchsize )  # -lambdaa*abs(w1)/w1

                b1 = b1 - lr * (np.mean( reluderw2, keepdims=True, axis=0))
            row_maxes = pred.max(axis=1).reshape(-1, 1)
            pred[:] = np.where(pred == row_maxes, 1, 0)
            result = np.all(pred == mnistytrain, axis=1)

            trainingepochaccuracy += np.sum(1 * result, axis=0)

            now = time.time()

            #Training Loss
        traininglosstable = np.append(traininglosstable, losssum/repet)




        #Testing Loss and Accuracy
        layer1testoutput = np.dot(testdatasetx, w1)
        layer1testoutputrelu = relu(layer1testoutput)
        #print(layer1outputrelu)

        predtest = softmax(np.dot(layer1testoutputrelu, w2))
        losstest = cceforward(predtest, testdatasety, w1, w2, reg=losstype, lambdaa=lambdaa)



        row_maxes = predtest.max(axis=1).reshape(-1, 1)
        predtest[:] = np.where(predtest == row_maxes, 1, 0)


        resulttest = np.all(predtest == testdatasety, axis=1)
        testingepochaccuracy += np.sum(1 * resulttest, axis=0)





        print(i+1, "th epoch:")
        print(i+1,"th epoch took: ", now - beforeepoch, " seconds")
        now = 0
        beforeepoch = 0

        print("Training Epoch Accuracy/Testing Epoch Accuracy:", (trainingepochaccuracy / howmanyimagesfortraining),"-",testingepochaccuracy / 12000)
        print("Training Loss/Testing Loss:", losssum/repet,"-",losstest)

        testinglosstable = np.append(testinglosstable, losstest)
        traininglosstable = np.append(traininglosstable,loss)
        trainingepochtable = np.append(trainingepochtable,trainingepochaccuracy/howmanyimagesfortraining)
        testingepochtable = np.append(testingepochtable,testingepochaccuracy/12000)
        loss = 0
        losstest = 0
        losssum = 0



    #np.save("weight1fornnl1.npy",w1)
    np.save("weight2fornnl1.npy",w2)
    #np.save("bias1fornnl1.npy",b1)
    #np.save("bias2fornnl1.npy",b2)
    #np.save("1weight1.npy", w1)
    return traininglosstable, testinglosstable, trainingepochtable, testingepochtable

"""
    np.save("weight1fornnl2.npy",w1)
    np.save("weight2fornnl2.npy",w2)
    np.save("bias1fornnl2.npy",b1)
    np.save("bias2fornnl2.npy",b2)
    #np.save("1weight1.npy", w1)

"""


    #np.save("1bias1.npy", b1)
    #np.save("1weight2.npy", w2)
    #np.save("1bias2.npy", b2)
    # np.save("bias.npy",b)













# Train
learningrate = 1e-3
traininglosstable, testinglosstable, trainingepochtable, testingepochtable = train(losstype="CCEL2",lr=learningrate, epoch= 10,
                                                batchsize=32, weightdecay = 1,weightdecayperiod= 10000,
                                                randomweights = True,node=784)


import matplotlib.pyplot as plt


print(traininglosstable)
plt.figure(figsize=(16,8))
titleforgraph = "Neural Network : CCE, lr = %0.02f, epoch = 10, "%learningrate
plt.title(titleforgraph)
plt.subplot(2,2,1)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.annotate('%0.2f' % traininglosstable[-1], xy=(1, traininglosstable[-1]), xytext=(2, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=traininglosstable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(traininglosstable)),traininglosstable)




plt.subplot(2,2,2)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.annotate('%0.2f' % testinglosstable[-1], xy=(1, testinglosstable[-1]), xytext=(2, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=testinglosstable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(testinglosstable)),testinglosstable)

plt.subplot(2,2,3)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.annotate('%0.2f' % trainingepochtable[-1], xy=(1, trainingepochtable[-1]), xytext=(2, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=trainingepochtable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(trainingepochtable)),trainingepochtable)

plt.subplot(2,2,4)
plt.title("Validation Accuracy") #Validation
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.annotate('%0.2f' % testingepochtable[-1], xy=(1, testingepochtable[-1]), xytext=(2, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=testingepochtable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(testingepochtable)),testingepochtable)

plt.savefig("nn1e-4lambda1e-5L2")
plt.show()






























"""

# Weight init ---
w1 = np.random.normal(0.5,0.1,(784,784))
w2 = np.random.normal(0.5,0.1,(784,10))
#w = np.load("weight.npy")

b1 = np.random.normal(0.5,0.1,(1,784))
b2 = np.random.normal(0.5,0.1,(1,10))

b1 = np.random.normal(0.5,0.1,(1,784))
b2 = np.random.normal(0.5,0.1,(1,10))

w1 = np.load("1weight1.npy")
w2 = np.load("1weight2.npy")

b2 = np.load("1bias2.npy")
b1 = np.load("1bias1.npy")

"""
