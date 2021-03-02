import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data_pickle
from neural import Neural
from test import test
from hyperparameter import *


def main():
    training_data, testing_data, label_dict = load_data_pickle()
    # inversing key,value for ease
    label_dict = {v:k for k, v in label_dict.items()}

    neural_net = Neural(training_data, hidden_layer)
    neural_net.parameter_init()
    # m is the number of images
    m = len(neural_net.X[1])

    print("Training data........")
    for i in range(1, epochs+1):
        cost_list = []
        for batch_no in range(int(m/batch_size)):
            X, y = neural_net.mini_batches(batch_size, batch_no)
            # al= activation of last layer
            al = neural_net.forward_propagation(X,y)
            J = neural_net.cost_function(al,y)
            cost_list.append(J)
            dw, db = neural_net.backpropagation(al,y)
            W, b = neural_net.update_parameter(alpha, dw, db)
        if i % 10 == 0:
            neural_test = test(testing_data, W, b, i)
        print("The cost of epoch {} is {}".format(i, np.sum(cost_list)))
        # plt.plot(cost_list)
        # plt.xlabel("epochs")
        # plt.ylabel("cost")
        # plt.show()

    return neural_net, neural_test


if __name__ == '__main__':
    neural, neural_test= main()

