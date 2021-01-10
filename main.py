from load_data import load_data
from neural import Neural


def main():
    label_dict_train, X_train, y_train = load_data('./DevanagariHandwrittenCharacterDataset/Train')
    label_dict_test, X_test, y_test = load_data('./DevanagariHandwrittenCharacterDataset/Test')
    neural = Neural(X_train, y_train, X_test, y_test, label_dict_train)
    return neural


if __name__ == '__main__':
    neural = main()
