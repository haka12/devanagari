import numpy as np
import random
import matplotlib.pyplot as plt


class Neural:
    def __init__(self, data, hidden_layer):
        # transforming data to put features as rows
        self.X = data[0].T
        self.y = data[1].T
        self.m = len(self.X[1])
        # making a list of all layers
        self.layers = [self.X.shape[0]] + hidden_layer + [self.y.shape[0]]
        # plt.imshow(self.train_X[:, 0].reshape(32, 32)) # checking if transformation is correct
        self.W = {}
        self.b = {}
        self.z = {}

    def parameter_init(self):
        np.random.seed(2)
        # random weight initialization with mean 0 and std deviation 0.01
        for l in range(1, len(self.layers)):
            self.W[l] = np.random.rand(self.layers[l], self.layers[l - 1]) * 0.01
            self.b[l] = np.zeros((self.layers[l], 1))

    def forward_propagation(self):
        cache = {}
        a = self.X
        for l in range(1, len(self.layers)):
            # z = w*a + b
            self.z[l] = np.dot(self.W[l], a) + self.b[l]
            # caching for use in backpropagation
            cache[l] = a
            # activation
            a = sigmoid(self.z[l])
            assert a.shape == (self.W[l].shape[0], a.shape[1])
        assert a.shape == (self.y.shape[0], self.X.shape[1])
        return a, cache

    def cost_function(self, pred):
        # cross entropy cost function
        J = np.sum((np.multiply(self.y, np.log(pred))) + (np.multiply((1 - self.y), np.log(1 - pred)))) / - self.m
        return J

    def backpropagation(self, cache, al):
        dw = {}
        db = {}
        # dz for layer l
        dz = al - self.y
        for l in range(len(self.layers) - 1, 0, -1):
            a = cache[l]
            dw[l] = 1 / self.m * np.dot(dz, a.T)
            db[l] = 1 / self.m * np.sum(dz, axis=1, keepdims=True)
            assert dw[l].shape == self.W[l].shape
            assert db[l].shape == self.b[l].shape
            # dz for layer l-1
            if l - 1 != 0:  # since z[0] doesnt exist and is not required
                dz = np.dot(self.W[l].T, dz) * sigmoid_deri(self.z[l - 1])
        return dw, db


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deri(z):
    return sigmoid(z) * (1 - sigmoid(z))
