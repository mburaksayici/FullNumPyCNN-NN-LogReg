import mnist
import numpy as np
import matplotlib.pyplot as plt
from forward2 import *




x_train, y_train, x_test, y_test = mnist.load()
x_train = x_train.reshape(-1,28,28)









numbofimages = 60000
mnistx = x_train/255
mnisty = y_train.reshape(1,numbofimages)
b = np.zeros((numbofimages,10)) # 10 is number of classes
b[np.arange(numbofimages), mnisty] = 1
mnisty = b


weights = np.random.random((2,3,3))
epoch = 1
x = 0
for i in range(epoch):
    x = x+1
    beg = (0 + x*20)%60000
    end = (beg + (x+1)*20)%60000


    conv = Conv(x,1,weights)

    a = conv.ForwardPass()




print(weights)

print(x[1])

print(a)