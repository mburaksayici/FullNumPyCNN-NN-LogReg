import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mnist
from numba import jit


#Data Prep---
x_train, y_train, x_test, y_test = mnist.load()
x_train = x_train.reshape(-1,28,28)
mnistx = x_train/255


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


w = np.load("cnnweight.npy")
print(w.shape)

weight = w.reshape(20,3,3)


print(mnistx[0].shape)
featuremap = ForwardConv(mnistx[0].reshape(-1,28,28),stride=2,weights=weight,flatten=False)





for i in range(20):
    plt.imshow(featuremap[i],cmap = matplotlib.cm.binary,interpolation = "nearest") #matplotlib.cm.binary
    plt.show()
