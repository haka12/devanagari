from neural import Neural
from hyperparameter import *


def test(testing_data, W, b):
    print("Testing......")
    neural_test = Neural(testing_data, hidden_layer)
    test_prediction, cache = neural_test.forward_propagation(W, b)
    J_test = neural_test.cost_function(test_prediction)
    print("The error on test set is", J_test)
    return test_prediction
