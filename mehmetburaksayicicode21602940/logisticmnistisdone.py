import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import random
np.random.seed(42)



#Data Prep---
x_train, y_train, x_test, y_test = mnist.load()
x_train = x_train.reshape(-1,28*28)

#Normalizing train
numbofimages = 60000
mnistx = x_train/255
#One hot encoding for train
mnisty = y_train.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnisty] = 1
mnisty = b
#---

#Normalizing
mnistxtest = x_test/255
numbofimages = 10000
#One hot encoding for train
mnistytest = y_test.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnistytest] = 1
mnistytest = b


mnistxtest = mnistxtest[0:12000]
mnistytest = mnistytest[0:12000]


# Weight init ---
w = np.random.normal(0,1,(784,10))



b = np.random.normal(0,1,(1,10))
#---




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)








def cceforward(x,y,w,reg,lambdaa):
    if reg == "CCE":

        return np.sum(-y * np.log(x +  1e-15) / x.shape[0]) # x + 1e-15

    elif  reg == "CCEL2":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0])+ lambdaa*np.sum(np.square(w))
    elif reg == "CCEL1":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0]) + lambdaa*np.sum(np.abs(w))













epoch = 10
traininglosstable = np.array([])
testinglosstable = np.array([])
trainingepochtable = np.array([])
testingepochtable = np.array([])

for i in range(epoch):

    losstype = "CCE"
    lr = 1e-2

    howmanyimagesfortraining = 60000
    batchsize = 32
    repet = int(howmanyimagesfortraining / batchsize)






    learningdecayperiod = 10
    learningdecay = 1
    losssum = 0
  
    beforeepoch = time.time()

    if i % learningdecayperiod == 0 and i > 1:
        lr = lr * learningdecay

    trainingepochaccuracy = 0
    testingepochaccuracy = 0

    lambdaa = 1e-2 # 1e-2 l2 için mthiş





    for a in range(repet):


        mnistxtrain = mnistx[batchsize * a:(batchsize * (a + 1))]
        mnistytrain = mnisty[batchsize * a:(batchsize * (a + 1))]

        pred = softmax(np.dot(mnistxtrain, w)+b)


        loss = cceforward(pred,mnistytrain,w=w,reg=losstype,lambdaa=lambdaa)

        if losstype == "CCEL1":

            w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain)) - lambdaa*np.abs(w)/w  # L1
            b = b - lr*(np.mean(pred-mnistytrain,axis=0,keepdims=True))


        elif losstype == "CCEL2":
            w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain)) -lambdaa*w  # L2
            b = b - lr*(np.mean(pred-mnistytrain,axis=0,keepdims=True))

        else:
            w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain))
            b = b - lr*(np.mean(pred-mnistytrain,axis=0,keepdims=True))


        losssum = losssum + loss

        row_maxes = pred.max(axis=1).reshape(-1, 1)
        pred[:] = np.where(pred == row_maxes, 1, 0)
        result = np.all(pred == mnistytrain, axis=1)

        trainingepochaccuracy += np.sum(1 * result, axis=0)

        now = time.time()

#        epochaccuracy += np.sum(1*result,axis=0)






        # Testing Loss and Accuracy


    predtest = softmax(np.dot(mnistxtest, w)+b)
    
    losstest = cceforward(predtest,mnistytest, w, reg=losstype, lambdaa=lambdaa)
    
    row_maxes = predtest.max(axis=1).reshape(-1, 1)
    predtest[:] = np.where(predtest == row_maxes, 1, 0)
    
    resulttest = np.all(predtest == mnistytest, axis=1)
    testingepochaccuracy += np.sum(1 * resulttest, axis=0)
        
    print(i, "th epoch:")
    print(i, "th epoch took: ", now - beforeepoch, " seconds")
    now = 0
    beforeepoch = 0

    print("Training Epoch Accuracy/Testing Epoch Accuracy:", (trainingepochaccuracy / howmanyimagesfortraining), "-",
          testingepochaccuracy / 10000)
    print("Training Loss/Testing Loss:", loss, "-", losstest)

    testinglosstable = np.append(testinglosstable, losstest)
    traininglosstable = np.append(traininglosstable, loss)
    trainingepochtable = np.append(trainingepochtable, trainingepochaccuracy / howmanyimagesfortraining)
    testingepochtable = np.append(testingepochtable, testingepochaccuracy / 10000)
    loss = 0
    losstest = 0
    losssum = 0



np.save("multilogcce",w)
#np.save("blogisticbestl1",b)





import matplotlib.pyplot as plt


print(traininglosstable)
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.annotate('%0.2f' % traininglosstable[-1], xy=(1, traininglosstable[-1]), xytext=(2, 0),
             xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=traininglosstable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(traininglosstable)),traininglosstable)




plt.subplot(2,2,2)
plt.title("Testing Loss")
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
plt.title("Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.annotate('%0.2f' % testingepochtable[-1], xy=(1, testingepochtable[-1]), xytext=(2, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.axhline(y=testingepochtable[-1], color='y', linestyle='-.')
plt.plot(np.arange(len(testingepochtable)),testingepochtable)
plt.savefig("logl2bestvis")
plt.show()

