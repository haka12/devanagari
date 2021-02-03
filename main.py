import matplotlib.pyplot as plt

from load_data import load_data_pickle
from neural import Neural
from test import test
from hyperparameter import *


def main():
    training_data, testing_data, data_label = load_data_pickle()
    # inversing key,value for ease
    data_label = {v:k for k,v in data_label.items()}
    neural_net = Neural(training_data, hidden_layer,data_label)
    neural_net.parameter_init()
    cost_list = []
    print("Training data........")
    for i in range(1, epochs):
        # al= activation of last layer
        al = neural_net.forward_propagation()
        J = neural_net.cost_function(al)
        print("The cost of epoch {} is {}".format(i,J))
        cost_list.append(J)
        dw, db = neural_net.backpropagation(al)
        W, b = neural_net.update_parameter(alpha, dw, db)
        # print(al)
    # plt.plot(cost_list)
    # plt.xlabel("epochs")
    # plt.ylabel("cost")
    # plt.show()
    test_prediction = test(testing_data, W, b,data_label)
    return neural_net, test_prediction,al,data_label


if __name__ == '__main__':
    neural, test_prediction,train_prediction,data_label= main()

