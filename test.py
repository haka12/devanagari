from neural import Neural
from hyperparameter import *
import numpy as np


def test(testing_data, W, b,iteration):
    print("Testing......")
    neural_test = Neural(testing_data, hidden_layer)
    test_prediction = neural_test.forward_propagation(neural_test.X, neural_test.y, W, b)
    test_prediction = np.argmax(test_prediction, axis = 0)
    test_y = np.argmax(neural_test.y, axis=0)
    correct = np.sum(test_prediction == test_y)
    accuracy = correct/neural_test.m * 100
    print("Total correct test prediction after {} iteration  is {} out of {} and accuracy is {}%".format(iteration, correct,neural_test.m, accuracy))
