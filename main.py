from load_data import load_data_pickle
from neural import Neural


def main():
    training_data, testing_data, data_label = load_data_pickle()
    train_X = training_data[0]
    train_y = training_data[1]
    test_X = testing_data[0]
    test_y = testing_data[1]
    # print(data_label)
    neural = Neural(train_X, train_y, test_X, test_y)
    a1, a2 = neural.forward_propagation()
    return neural, a1, a2


if __name__ == '__main__':
    neural, a1, a2 = main()
