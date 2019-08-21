import mnist
import numpy as np
import matplotlib.pyplot as plt
from forward2 import *
import matplotlib
wl1 = np.load("multilogccel1.npy")


wl2 = np.load("multilogccel2.npy")
w = np.load("multilog.npy")


w = w.reshape(28,28,10)
wl2 = wl2.reshape(28,28,10)
wl1 = wl1.reshape(28,28,10)




plt.figure(figsize=(16,8))

for i in range(10):
    print(i)

    titl = str("W NoReg, %d"%i)

    plt.subplot(1, 10,i+1)
    plt.title(titl)
    plt.imshow(w[:, :, i], cmap=matplotlib.cm.binary, interpolation="nearest")
plt.show()
plt.figure(figsize=(16,8))

for i in range(10):
    print(i)
    plt.subplot(1,10,i+1)
    titl = str("W with l1, %d"%i)
    plt.title(titl)
    plt.imshow(wl1[:, :, i], cmap=matplotlib.cm.binary, interpolation="nearest")
plt.show()
plt.figure(figsize=(16,8))

for i in range(10):
    print(i)
    plt.subplot(1,10,i+1)
    titl = str("W with l2, %d"%i)
    plt.title(titl)
    plt.imshow(wl2[:, :, i], cmap=matplotlib.cm.binary, interpolation="nearest")
plt.show()





"""

for i in range(10):
    plt.imshow(w[:,:,i],cmap = matplotlib.cm.binary,interpolation = "nearest") #matplotlib.cm.binary
    plt.show()
"""