import numpy as np
import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import cython




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










def relu(x):
    x.copy()
    out = x*(x > 0)
    return out




def cceforward(x,y,w1,w2,lambdaa = 1,reg = "CCE",):
    if reg == "CCE":
        return np.sum(-y * np.log(x + 1e-15) / x.shape[0])
    elif  reg == "CCEL2":
        return np.sum(-y * np.log(x  + 0) / x.shape[0]) + lambdaa*np.sum(np.square(w1))
    elif reg == "CCEL1":
        return np.sum(-y * np.log(x  + 0) / x.shape[0]) + lambdaa*np.sum(np.abs(w1))  +lambdaa*np.sum(np.abs(w2))




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)




def ForwardConv(inputarray, stride, weights, flatten = True):




    cdef int filternumber = weights.shape[0]



    cdef int height = weights.shape[1]

    cdef int width = weights.shape[2]

    cdef int numbofimagesinputarray = inputarray.shape[0]

    cdef int hoffmap = inputarray[0].shape[0]  # hoffmap -> height of feature map


    cdef int woffmap = inputarray[0].shape[1]  # woffmap -> width of feature map

    cdef int widthtour = int(1 + (woffmap - width) / stride)

    cdef int heighttour = int(1 + (hoffmap - height) / stride)
    cdef int i,j,k,l = 0
    # print(widthtour)
    # print(heighttour)
    featuremap = np.zeros((inputarray.shape[0], filternumber, widthtour, heighttour))

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






def BackwardConvAutoDiff(inputarray, stride, weights,derivweights,lr):
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

                    weights[j] = weights[j]-lr*array_*np.sum(derivweights[i,j,:,:])


    return weights
























def BackwardConv(inputarray, stride, weights,derivweights,lr):
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

                    weights[j] = weights[j]-lr*array_*np.sum(derivweights[i,j,:,:])


    return weights




def train(losstype="CCE",lr=0.001, epoch= 20,  batchsize=32, weightdecay = 1,weightdecayperiod= 10,node=10,
          testdatasetx=mnistx[48000:].reshape(-1,28,28),testdatasety = mnisty[48000:]):

    traininglosstable = np.array([])
    testinglosstable = np.array([])
    trainingepochtable = np.array([])
    testingepochtable = np.array([])

    weightcnn1 = np.random.normal(0, 0.01, (20, 3, 3))
    weightfcn = np.random.normal(0, 0.01, (3380, 10))
    bfcn = np.random.normal(0, 0.2, (1, 10))

    for i in range(epoch):
        beforeepoch = time.time()

        if i % weightdecayperiod == 0 and i > 1:
            lr = lr * weightdecay

        trainingepochaccuracy = 0
        testingepochaccuracy = 0
        lambdaa = 1e-5
        howmanyimagesfortraining = 48000 # 60000
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

            loss = cceforward(pred, mnistytrain, w1=0, w2=0, reg=losstype, lambdaa=lambdaa)

            losssum = losssum + loss
            if losstype == "CCE":

                weightfcn = weightfcn - lr * np.dot(cnnout.T, predminusy)
                derivuntilcnn = np.dot(weightfcntemp, predminusy.T).reshape(-1, 20, 13, 13)

                weightcnn1 = BackwardConv(mnistxtrain, 2, weightcnn1, derivuntilcnn, lr)

            traininglosstable = np.append(traininglosstable, losssum)



            row_maxes = pred.max(axis=1).reshape(-1, 1)
            pred[:] = np.where(pred == row_maxes, 1, 0)
            result = np.all(pred == mnistytrain, axis=1)

            trainingepochaccuracy += np.sum(1 * result, axis=0)


            print(100 * a / repet, "% of Epoch is finished")
            print("Batch Accuracy:",np.sum(1 * result, axis=0)/32)


            now = time.time()

        cnnout = ForwardConv(testdatasetx, 2, weightcnn1, flatten=True)
        # relucnnout = relu(cnnout)
        cnntopred = np.dot(cnnout, weightfcn)  # + bfcn
        predtest = softmax(cnntopred)


        losstest = cceforward(predtest, testdatasety, w1=0, w2=0, reg=losstype, lambdaa=lambdaa)



        row_maxes = predtest.max(axis=1).reshape(-1, 1)
        predtest[:] = np.where(predtest == row_maxes, 1, 0)


        resulttest = np.all(predtest == testdatasety, axis=1)
        testingepochaccuracy += np.sum(1 * resulttest, axis=0)

        print(i+1, "th epoch:")
        print(i+1,"th epoch took: ", now - beforeepoch, " seconds")
        now = 0
        beforeepoch = 0

        print("Training Epoch Accuracy/Testing Epoch Accuracy:", (trainingepochaccuracy / howmanyimagesfortraining),"-",testingepochaccuracy / 12000)
        print("Training Loss/Testing Loss:", loss,"-",losstest)

        testinglosstable = np.append(testinglosstable, losstest)
        traininglosstable = np.append(traininglosstable,loss)
        trainingepochtable = np.append(trainingepochtable,trainingepochaccuracy/howmanyimagesfortraining)
        testingepochtable = np.append(testingepochtable,testingepochaccuracy/12000)
        loss = 0
        losstest = 0






    print("Epoch : ",epoch)



train()




"""
b = ForwardConv(a,2,weightcnn2)

print(b.shape,"b")


np.convolve(mnistx[0],)


plt.imshow(mnistx[0].reshape(28,28))
plt.show()
plt.imshow(a[0])
plt.show()
plt.imshow(b[0])
plt.show()
"""