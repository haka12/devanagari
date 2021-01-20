import numpy as np
import random
import matplotlib.pyplot as plt


class Neural:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.X = train_X.T
        self.y = train_y.T
        # plt.imshow(self.X[:, 0].reshape(32, 32))
        self.n_x = self.X.shape[0]  # size of input layer
        self.n_h = 100  # size of hidden layer
        self.n_y = self.y.shape[0]  # size of output layer
        np.random.seed(2)
        # random weight initialization with mean 0 and std deviation 0.01
        self.W1 = np.random.rand(self.n_h, self.n_x) * 0.01
        self.W2 = np.random.rand(self.n_y, self.n_h) * 0.01
        # bias initialized as 0
        self.b1 = np.zeros((self.n_h, 1))
        self.b2 = np.zeros((self.n_y, 1))

    def parameter_init(self):
        pass
    def forward_propagation(self):
        z1 = np.dot(self.W1, self.X) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = sigmoid(z2)
        assert (a2.shape == (self.n_y, self.X.shape[1]))
        return a1,a2
        # cache =


def sigmoid_deri(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
