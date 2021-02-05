import matplotlib.image as img
import os
import pickle
import numpy as np


def load_data(path):
    images = []
    no_of_data = []

    # encoding the data with labels
    label_dict = {key: encoding for key, encoding in zip(os.listdir(path), [*range(1, len(os.listdir(path)) + 1)])}

    # walking through the folders to read each image
    for subdirs, _, files in os.walk(path):
        images = images + [img.imread(subdirs + os.sep + file) for file in files]
        # reading the no. of files in each directory
        if len(files) != 0:
            no_of_data.append(len(files))

    # converting the list into ndarray
    X_values = np.array(images)
    # making an array of encoded labels
    for values in label_dict.values():
        y_value = values * np.ones((no_of_data[values - 1], 1))
        if values > 1:
            y_value = np.append(temp, y_value)
        temp = y_value

    # vectorizing the y_value
    y_value = [vectorize(int(y), len(label_dict.values())) for y in y_value]
    y_value = np.array(y_value)
    # removing a single dimensional value
    y_value = np.squeeze(y_value)
    return label_dict, X_values, y_value


def vectorize(y, l):
    v = np.zeros((l, 1))
    v[y - 1] = 1
    return v


def pickle_dump():
    label_dict_train, X_train, y_train = load_data('./Data/Train')
    label_dict_test, X_test, y_test = load_data('./Data/Test')
    train_length = len(X_train)
    test_length = len(X_test)
    training_data = X_train.reshape(train_length, -1), y_train
    testing_data = X_test.reshape(test_length, -1), y_test
    data_set = [training_data, testing_data, label_dict_train]
    with open('data.pkl.gz', 'wb') as f:
        pickle.dump(data_set, f)


def load_data_pickle():
    if not os.path.isfile('./data.pkl.gz'):
        print("creating serialized data ")
        pickle_dump()
    with open('./data.pkl.gz', 'rb') as f:
        return pickle.load(f)
