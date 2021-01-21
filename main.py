from load_data import load_data_pickle
from neural import Neural


def main():
    training_data, testing_data, data_label = load_data_pickle()
    # print(data_label)
    hidden_layer = [150, 50]
    neural_net = Neural(training_data, hidden_layer)
    neural_net.parameter_init()
    a, cache = neural_net.forward_propagation()
    J = neural_net.cost_function(a)
    return neural_net, a, cache,J


if __name__ == '__main__':
    neural, a, cache,J = main()
