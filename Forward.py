import numpy as np


class Conv():
    import numpy as np
    def __init__(self, inputarray, stride, weights):
        self.filternumber = weights.shape[0]
        self.width = weights.shape[2]
        self.height = weights.shape[1]
        self.stride = stride
        self.inputarray = inputarray
        self.weights = weights

    def ForwardPass(self):
        print("Input Array Shape for Conv : ",self.inputarray.shape)
        self.hoffmap = self.inputarray[0].shape[0]  # hoffmap -> height of feature map

        self.woffmap = self.inputarray[0].shape[1]  # woffmap -> width of feature map

        self.widthtour = int(1 + (self.woffmap - self.width) / self.stride)
        self.heighttour = int(1 + (self.hoffmap - self.height) / self.stride)
        # print(self.widthtour)
        # print(self.heighttour)
        self.featuremap = np.zeros((self.inputarray.shape[0], self.filternumber, self.widthtour, self.heighttour))

        for i in range(self.inputarray.shape[0]):
            for j in range(self.filternumber):
                for k in range(int(self.heighttour)):
                    for l in range(int(self.widthtour)):
                        # if channel is implemented  we need another for loop for channel
                        # print(self.stride*l,self.width+self.stride*(l),"---")
                        self.array_ = self.inputarray[i][self.stride * k:self.height + self.stride * k,
                                      self.stride * l:self.width + self.stride * l]

                        # print(self.inputarray[i][l:self.height,self.width*l:self.width*(l+1)].shape)
                        #print(i, j, k, l)
                        self.featuremap[i, j, k, l] = np.sum(np.multiply(self.array_, self.weights[j]))

        return self.featuremap




class FCN():
    def __init__(self, inputarray):
        self.inputarray = inputarray
        self.lossoutput = 0

    def ForwardPass(self,fcnweights):
        self.inputarray = self.inputarray.reshape(self.inputarray.shape[0], -1)

        # print(self.inputarray.shape,"beforeforward shape")
        #print(self.inputarray.dot(fcnweights))
        return self.inputarray.dot(fcnweights)

    def backwardpass(self, lastderivative,fcnweights):
        dfcn = lastderivative.dot(self.inputarray)
        newweights = fcnweights - dfcn.T * 0.2


        #print(fcnweights.shape, newweights.shape)

        return dfcn, newweights








class Activation():
    def __init__(self, actstring, inputarray):

        self.activation = actstring
        self.inputarray = inputarray

    def relu(self):
        print("input array shape",self.inputarray.shape)
        self.inputarray = np.maximum(self.inputarray, 0)

        return self.inputarray

    def softmax(self):
        """
        print("out shape ",out.shape)
        print("exp shape softmax : ", np.exp(self.inputarray))
        print("Sum Shape sotftmax: ", np.sum(np.exp(self.inputarray),axis=1,keepdims=True))
        print("outofsoftmax : ", np.exp(self.inputarray)/np.sum(np.exp(self.inputarray),axis=1,keepdims=True))
        """
        return np.exp(self.inputarray) / np.sum(np.exp(self.inputarray), axis=1, keepdims=True)

    def ForwardPass(self):
        if self.activation == "relu":
            return self.relu()
        elif self.activation == "softmax":
            return self.softmax()


    def backwardpass(self,lastderivative):
        if self.activation == "softmax":
            #outsoftmax = self.softmax()
            #outsoftmax = outsoftmax.reshape(-1, 1)
            #
            #
            return lastderivative   #np.diagflat(outsoftmax) - np.dot(outsoftmax, outsoftmax.T)

        elif self.activation == "relu":
            lastderivative[lastderivative < 0] = 0
            return lastderivative


class Loss():
    def __init__(self, inputarray, y, loss):
        self.inputarray = inputarray
        self.y = y
        self.loss = loss

    def cceforward(self):
        return -self.y * np.log(self.inputarray + 1e-20) / self.inputarray.shape[0]  #+ 1e-004

    def ForwardPass(self):
        if self.loss == "CCE":
            return self.cceforward()

    def BackwardPass(self):

        if self.loss == "CCE":

            #print(self.inputarray.shape)
            ##print("-")
            #print(self.y.shape)
            ##print("----")
            ##dloss =  -self.y/self.inputarray
            #dloss = np.dot(self.y,self.inputarray.T)
            #
            #dloss[dloss == np.inf] = 0
            #print(dloss)
            return  (self.inputarray-self.y).T


