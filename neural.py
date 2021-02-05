import numpy as np
import matplotlib.pyplot as plt


class Neural:
    def __init__(self, data, hidden_layer):
        # transforming data to put features as rows
        self.X = data[0].T
        self.y = data[1].T
        self.m = len(self.X[1])
        # shuffling data
        np.random.seed(1)
        per = np.random.permutation(len(self.X[1]))
        self.X = self.X[:, per]
        self.y = self.y[:, per]
        # making a list of all layers
        self.layers = [self.X.shape[0]] + hidden_layer + [self.y.shape[0]]
        # initializing w,b,z,a
        self.W = {}
        self.b = {}
        self.z = {}
        self.a = {}
        # print(data_label[np.argmax(self.y[:, 2])+1])
        # plt.imshow(self.X[:, 2].reshape(32, 32))  # checking if shuffling  is correct

    def parameter_init(self):
        np.random.seed(4)
        # random weight initialization with mean 0 and std deviation 0.01
        for l in range(1, len(self.layers)):
            self.W[l] = np.random.rand(self.layers[l], self.layers[l - 1]) * 0.01
            self.b[l] = np.zeros((self.layers[l], 1))

    def forward_propagation(self, *args):
        # args is here for test set
        if len(args):
            W = args[0]
            b = args[1]
        else:
            W = self.W
            b = self.b
        cache = {}
        self.a[0] = self.X
        for l in range(1, len(self.layers)):
            # z = w*a + b
            self.z[l] = np.dot(W[l], self.a[l - 1]) + b[l]
            # activation
            if l == len(self.layers):
                self.a[l] = softmax(self.z[l])
            else:
                self.a[l] = sigmoid(self.z[l])
        assert self.a[l].shape == (self.y.shape[0], self.X.shape[1])
        return self.a[l]

    def cost_function(self, pred):
        # cross entropy cost function
        J = np.sum((np.multiply(self.y, np.log(pred))) + (np.multiply((1 - self.y), np.log(1 - pred)))) / - self.m
        return J

    def backpropagation(self, al):
        dw = {}
        db = {}
        # dz for layer l(dz is cost derivative wrt z)
        dz = al - self.y
        for l in range(len(self.layers) - 1, 0, -1):
            # dw = cost derivative wrt w
            dw[l] = np.dot(dz, self.a[l - 1].T) / self.m
            db[l] = np.sum(dz, axis=1, keepdims=True) / self.m
            assert dw[l].shape == self.W[l].shape
            assert db[l].shape == self.b[l].shape
            # dz for layer l-1
            if l - 1 != 0:  # since z[0] doesnt exist and is not required
                _ = np.dot(self.W[l].T, dz)
                dz = np.multiply(_, sigmoid_deri(self.z[l - 1]))
        return dw, db

    def update_parameter(self, alpha, dw, db):
        for l in range(1, len(self.layers)):
            self.W[l] = self.W[l] - alpha * dw[l]
            self.b[l] = self.b[l] - alpha * db[l]
        return self.W,self.b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deri(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    return np.exp(z)/sum(np.exp(z))
