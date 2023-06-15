import numpy as np


class DatasetLoader:
    def __init__(self, dataset, multiclass_flag=False):
        self.dataset = dataset
        self.multiclass_flag = multiclass_flag

    def run(self):
        return self.multiclass_flag, self.dataset.data, self.dataset.target


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 - np.exp(-z))

def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)

