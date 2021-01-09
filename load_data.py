import matplotlib.image as img
import os
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
    return label_dict, X_values, y_value


label_dict_train, X_train, y_train = load_data('./DevanagariHandwrittenCharacterDataset/Train')
label_dict_test, X_test, y_test = load_data('./DevanagariHandwrittenCharacterDataset/Test')
print("Training labels", label_dict_train)
print("Testing labels", label_dict_test)
print("shape of X_train", X_train.shape)
print("shape of y_train", y_train.shape)
print("shape of X_test", X_test.shape)
print("shape of y_test", y_test.shape)
