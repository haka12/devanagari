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

    def parameter_init(self):
        np.random.seed(2)
        # random weight initialization with mean 0 and std deviation 0.01
        for l in range(1, len(self.layers)):
            self.W[l] = np.random.rand(self.layers[l], self.layers[l - 1]) * 0.01
            self.b[l] = np.zeros((self.layers[l], 1))

    def forward_propagation(self):
        a = self.X
        for l in range(1, len(self.layers)):
            # z = w*a + b
            z = np.dot(self.W[l], a) + self.b[l]
            # activation
            a = sigmoid(z)
            assert a.shape == (self.W[l].shape[0], a.shape[1])
        assert a.shape == (self.y.shape[0], self.X.shape[1])
        cache = self.W, self.b, a
        return a, cache

    def cost_function(self, pred):
        # cross entropy cost function
        J = np.sum((np.multiply(self.y, np.log(pred))) + (np.multiply((1-self.y), np.log(1 - pred)))) / - self.m
        return J


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deri(z):
    return sigmoid(z) * (1 - sigmoid(z))
