from neural import Neural
from hyperparameter import *
import numpy as np


def test(testing_data, W, b,data_label):
    print("Testing......")
    neural_test = Neural(testing_data, hidden_layer,data_label)
    test_prediction = neural_test.forward_propagation(W, b)
    J_test = neural_test.cost_function(test_prediction)
    # test_prediction = np.where(test_prediction == test_prediction.max(axis=0)[:, np.newaxis], 1, 0)
    # print("The error on test set is", J_test)
    return test_prediction
