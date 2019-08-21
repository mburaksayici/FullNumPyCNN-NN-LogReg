import mnist
import numpy as np
import matplotlib.pyplot as plt




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





"""
mnistxtrain = mnistx[0:300]
mnistytrain = mnisty[0:300]                                    
print(mnistxtrain.shape)            
"""

# Weight init ---
w = np.random.normal(0,0.01,(784,10))

#w = np.load("weight.npy")

b = np.random.normal(0,0.001,(1,10))
#---




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)




#pred = softmax(np.dot(mnistxtrain,w))


def cceforward(x,y,lambdaa = 1,reg = "CCE"):
    if reg == "CCE":
        return np.sum(-y * np.log(x + 1e-15) / x.shape[0])
    elif  reg == "CCEL2":

        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0]) + lambdaa*np.sum(np.square(w))
    elif reg == "CCEL1":
        return np.sum(-y * np.log(x  + 1e-15) / x.shape[0]) + lambdaa*np.sum(np.abs(w))

lossa = np.array([])
for i in range(20):

    lr = 1e-4
    epochaccuracy = 0
    lambdaa = 1e-2
    for a in range(200):


        mnistxtrain = mnistx[300*a:(300*(a+1))]
        mnistytrain = mnisty[300*a:(300*(a+1))]

        pred = softmax(np.dot(mnistxtrain, w)+b)


        loss = cceforward(pred, mnistytrain,reg ="CCEL2",lambdaa=lambdaa)

        #w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain))  # L2
        #b = b - lr*(pred-mnistytrain)



        w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain)) -lambdaa*w  # L2
        b = b - lr*(pred-mnistytrain)-lambdaa*b


        #w = w - lr*np.dot(mnistxtrain.T,(pred-mnistytrain)) -lambdaa*abs(w)/w  # LASSO
        #b = b + lr*(pred-mnistytrain)-lambdaa*abs(b)/b


        lossa = np.append(lossa,loss)

        row_maxes = pred.max(axis=1).reshape(-1, 1)
        pred[:] = np.where(pred == row_maxes, 1, 0)
        result = np.all(pred == mnistytrain, axis=1)

        #for lo in range(200):
        #    print(pred[lo],mnistytrain[lo])
        epochaccuracy += np.sum(1*result,axis=0)
    #lr = lr*0.94


    print(i,"th epoch:")
    print("Epoch Accuracy:",epochaccuracy/60000)
    print("Loss:",loss)

import matplotlib.pyplot as plt




plt.plot(np.arange(len(lossa)),lossa)
plt.show()




mnistxtest = x_test/255
numbofimages = 10000
mnistytest = y_test.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnistytest] = 1
mnistytest = b

np.save("weight.npy",w)
np.save("bias.npy",b)


pred = softmax(np.dot(mnistxtest, w))


row_maxes = pred.max(axis=1).reshape(-1, 1)
pred[:] = np.where(pred == row_maxes, 1, 0)
result = np.all(pred == mnistytest, axis=1)



print(np.sum(1*result)/10000)


print(w[0])