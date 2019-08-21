import numpy as np
import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit,njit,prange, optional
import numba




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


#weightcnn1 = np.random.normal(0,0.01,(20,3,3))
#weightfcn = np.random.normal(0,0.01,(3380,10))
#bfcn = np.random.normal(0,0.2,(1,10))









@jit
def relu(x):
    x.copy()
    out = x*(x > 0)
    return out


@jit
def cceforward(x,y,w1,lambdaa = 1,reg = "CCE"):
    if reg == "CCE":
        return np.sum(-y * np.log(x + 1e-15) / x.shape[0])
    elif  reg == "CCEL2":
        return np.sum(-y * np.log(x  + 1e-150) / x.shape[0]) + lambdaa*np.sum(np.square(w1))
    elif reg == "CCEL1":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0]) + lambdaa*np.sum(np.abs(w1))



@jit
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)




@jit
def ForwardConv(inputarray, stride, weights, flatten = True):

    filternumber = weights.shape[0]
    height = weights.shape[1]
    width = weights.shape[2]
    stride = stride
    numbofimagesinputarray = inputarray.shape[0]


    hoffmap = inputarray[0].shape[0]  # hoffmap -> height of feature map


    woffmap = inputarray[0].shape[1]  # woffmap -> width of feature map

    widthtour = int(1 + (woffmap - width) / stride)

    heighttour = int(1 + (hoffmap - height) / stride)
    # print(widthtour)
    # print(heighttour)
    featuremap = np.zeros((inputarray.shape[0], filternumber, widthtour, heighttour),dtype=np.float64)

    for i in range(inputarray.shape[0]):
        for j in range(filternumber):
            for k in range(int(heighttour)):
                for l in range(int(widthtour)):
                    # if channel is implemented  we need another for loop for channel
                    # print(stride*l,width+stride*(l),"---")
                    array_ = inputarray[i][stride * k:height + stride * k,
                                  stride * l:width + stride * l]


                    # print(inputarray[i][l:height,width*l:width*(l+1)].shape)
                    # print(i, j, k, l)
                    featuremap[i, j, k, l] = np.sum(np.multiply(array_, weights[j]))
    if flatten== False:
        return featuremap.reshape(-1,k+1,l+1)
    else:
        return featuremap.reshape(numbofimagesinputarray,-1)






















@jit
def BackwardConv(inputarray, stride, weights,derivweights,lr,losstype,lambdaa):
    filternumber = weights.shape[0]
    height = weights.shape[1]
    width = weights.shape[2]
    stride = stride
    numbofimagesinputarray = inputarray.shape[0]

    hoffmap = inputarray[0].shape[0]  # hoffmap -> height of feature map

    woffmap = inputarray[0].shape[1]  # woffmap -> width of feature map

    widthtour = int(1 + (woffmap - width) / stride)

    heighttour = int(1 + (hoffmap - height) / stride)
    # print(widthtour)
    # print(heighttour)
    featuremap = np.zeros((inputarray.shape[0], filternumber, widthtour, heighttour))

    for i in range(inputarray.shape[0]):
        for j in range(20):
            for k in range(int(heighttour)):
                for l in range(int(widthtour)):
                    # if channel is implemented  we need another for loop for channel
                    # print(stride*l,width+stride*(l),"---")
                    array_ = inputarray[i][stride * k:height + stride * k, stride * l:width + stride * l]


                    # print(inputarray[i][l:height,width*l:width*(l+1)].shape)
                    # print(i, j, k, l)
                    if losstype =="CCE":
                        weights[j] = weights[j]-lr*array_*np.sum(derivweights[i,j,:,:])
                    elif losstype == "CCEL2":
                        weights[j] = weights[j] - lr * array_ * np.sum(derivweights[i, j, :, :]) - lambdaa*weights[j]
                    elif losstype == "CCEL1":
                        weights[j] = weights[j] - lr * array_ * np.sum(derivweights[i, j, :, :]) - lambdaa *abs(weights[j])/weights[j]

    return weights



def train(losstype="CCE",lr=0.0001, epoch=10,  batchsize=32, weightdecay = 1,weightdecayperiod= 10,lambdaa=1e-5,
          testdatasetx=mnistx[480:].reshape(-1,28,28),testdatasety = mnisty[480:]):

    traininglosstable = np.array([])
    testinglosstable = np.array([])
    trainingepochtable = np.array([])
    testingepochtable = np.array([])

    weightcnn1 = np.random.normal(0, 0.01, (20, 3, 3))
    weightfcn = np.random.normal(0, 0.02432521277, (3380, 10))
    bfcn = np.random.normal(0, 0.2, (1, 10))

    for i in range(epoch):
        beforeepoch = time.time()

        if i % weightdecayperiod == 0 and i > 1:
            lr = lr * weightdecay

        trainingepochaccuracy = 0
        testingepochaccuracy = 0
        lambdaa = 1e-5
        howmanyimagesfortraining = 320 # 60000
        losssum = 0
        repet = int(howmanyimagesfortraining / batchsize)



        for a in range(repet):

            mnistxtrain = mnistx[batchsize * a:(batchsize * (a + 1))].reshape(-1,28,28)
            mnistytrain = mnisty[batchsize * a:(batchsize * (a + 1))]


            cnnout = ForwardConv(mnistxtrain, 2, weightcnn1, flatten=True)
            #relucnnout = relu(cnnout)
            cnntopred =  np.dot(cnnout,weightfcn) #+ bfcn
            pred = softmax(cnntopred)

            weightfcntemp = weightfcn.copy() # For grad
            predminusy = pred-mnistytrain # For grad

            loss = cceforward(pred, mnistytrain, w1=weightcnn1,  reg=losstype, lambdaa=lambdaa)

            losssum = losssum + loss

            weightfcn = weightfcn - lr * np.dot(cnnout.T, predminusy)
            derivuntilcnn = np.dot(weightfcntemp, predminusy.T).reshape(-1, 20, 13, 13)

            weightcnn1 = BackwardConv(mnistxtrain, 2, weightcnn1, derivuntilcnn, lr,losstype,lambdaa)

            traininglosstable = np.append(traininglosstable, losssum)





            row_maxes = pred.max(axis=1).reshape(-1, 1)
            pred[:] = np.where(pred == row_maxes, 1, 0)
            result = np.all(pred == mnistytrain, axis=1)

            trainingepochaccuracy += np.sum(1 * result, axis=0)


            print(100 * a / repet, "% of Epoch",i+1," is finished")
            print("Batch Accuracy:",np.sum(1 * result, axis=0)/32)

            now = time.time()

        cnnout = ForwardConv(testdatasetx, 2, weightcnn1, flatten=True)
        # relucnnout = relu(cnnout)
        cnntopred = np.dot(cnnout, weightfcn)  # + bfcn
        predtest = softmax(cnntopred)


        losstest = cceforward(predtest, testdatasety, w1=0, reg=losstype, lambdaa=lambdaa)



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

    np.save("cnnweightl1",weightcnn1)
    np.save("cnnfcnweightl1",weightfcn)
    return traininglosstable, testinglosstable, trainingepochtable, testingepochtable

    print("Epoch : ",epoch)



learningrate=1e-3
traininglosstable, testinglosstable, trainingepochtable, testingepochtable  = train(losstype="CCE",lr=learningrate, epoch=10,  batchsize=32, weightdecay = 1,weightdecayperiod= 10,
lambdaa=1e-6,testdatasetx=mnistx[48000:].reshape(-1,28,28),testdatasety = mnisty[48000:])


















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

plt.savefig("cnnNOLOSSlr1e-5")
plt.show()