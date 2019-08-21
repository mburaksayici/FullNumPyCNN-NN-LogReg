import mnist
import numpy as np
import matplotlib.pyplot as plt
from forward2 import *
import matplotlib
#w1l2 = np.load("weight1fornnl2.npy")
w2l2 = np.load("weight2fornnl2.npy")

#w1 = np.load("weight1fornn.npy")
w2 = np.load("weight2fornn.npy")

#w1l1 = np.load("weight1fornnl1.npy")
w2l1 = np.load("weight2fornnl1.npy")

import mnist
import numpy as np
import matplotlib.pyplot as plt



def relu(x):
    x.copy()
    out = x*(x > 0)
    return out


w2 = (w2.reshape(-1,28,28))
w2l2 = (w2l2.reshape(-1,28,28))
w2l1 = (w2l1.reshape(-1,28,28))


plt.figure(figsize=(16,8))

for i in range(9):
    print(i)

    titl = str("W2 NoReg, %d"%i)

    plt.subplot(1, 9,i+1)
    plt.title(titl)
    plt.imshow(w2[i],cmap="Greys")
plt.show()


plt.figure(figsize=(16,8))
for i in range(9):
    print(i)
    plt.subplot(1,9,i+1)
    titl = str("W2 with l2, %d"%i)
    plt.title(titl)
    plt.imshow(w2l2[i],cmap="Greys")
plt.show()
plt.figure(figsize=(16,8))


for i in range(9):
    print(i)
    plt.subplot(1,9,i+1)
    titl = str("W2 with l1, %d"%i)
    plt.title(titl)
    plt.imshow(w2l1[i],cmap="Greys")
plt.show()











# finding mean img
"""

# Data Prep---
x_train, y_train, x_test, y_test = mnist.load()
x_train = x_train.reshape(-1, 28 * 28)

numbofimages = 60000
mnistx = x_train / 255

mnisty = y_train.reshape(1, numbofimages)
"""

"""
b = np.zeros((numbofimages, 10))  # 10 is number of classes
b[np.arange(numbofimages), mnisty] = 1
mnisty = b
"""
"""

a,b= np.where(mnisty==2)

summ = np.array([])
for elements in b[0:200]:

    summ = np.append(summ,mnistx[elements])


summ = summ.reshape(-1,28,28)
print(summ.shape)
summean = summ.mean(axis=0)

plt.subplot(1,2,1)
plt.imshow(summean)
plt.subplot(1,2,2)

w2 = w2.reshape(-1,28,28)

w2[w2>0.5] =1

w2[w2<0.5] =0
plt.imshow(w2[2])
plt.show()
"""





