from load_data import load_data_pickle
from neural import Neural


def main():
    training_data, testing_data, data_label = load_data_pickle()
    # print(data_label)
    hidden_layer = [150, 50]
    neural_net = Neural(training_data, hidden_layer)
    neural_net.parameter_init()
    for i in range(100):
        # al= activation of last layer and cache ={a[l-1]}
        al, cache = neural_net.forward_propagation()
        J = neural_net.cost_function(al)
        dw, db = neural_net.backpropagation(cache, al)
        neural_net.update_parameter(0.05, dw, db)
    return neural_net, dw, db


if __name__ == '__main__':
    neural, dw, db = main()
