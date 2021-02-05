import matplotlib.pyplot as plt

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
    cost_list = []

    print("Training data........")
    for i in range(1, epochs+1):
        # al= activation of last layer
        al = neural_net.forward_propagation()
        J = neural_net.cost_function(al)
        print("The cost of epoch {} is {}".format(i,J))
        cost_list.append(J)
        dw, db = neural_net.backpropagation(al)
        W, b = neural_net.update_parameter(alpha, dw, db)
        if i % 100 == 0:
            neural_test = test(testing_data, W, b, i)
    # plt.plot(cost_list)
    # plt.xlabel("epochs")
    # plt.ylabel("cost")
    # plt.show()

    return neural_net, neural_test


if __name__ == '__main__':
    neural, neural_test= main()

