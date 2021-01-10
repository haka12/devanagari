import numpy as np
import random


class Neural:
    def __init__(self, X_train, y_train, X_test, y_test, label_dict):
        self.label_dict = label_dict
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

