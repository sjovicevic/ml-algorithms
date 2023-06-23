import numpy as np
import pandas as pd


class DatasetLoader:
    def __init__(self, dataset, multiclass_flag=False):
        self.dataset = dataset
        self.multiclass_flag = multiclass_flag

    def run(self):
        return self.multiclass_flag, self.dataset.data, self.dataset.target


def softmax(z, derivative=False):
    if derivative:
        return np.diagflat(z) - np.dot(z, z.T)
    else:
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def tanh(z, derivative=False):
    if derivative:
        pass
    else:
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)


def loss(a, output, n_samples):
    y_one_hot = np.array(pd.get_dummies(output, dtype='int8'))
    return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(a)))


def relu(z, derivative=False):
    if derivative:
        z = np.where(z < 0, 0, z)
        z = np.where(z >= 0, 1, z)
        return z
    return np.maximum(0, z)


def sigmoid(z, derivative=False):
    if derivative:
        return np.multiply(z, 1-z)
    return 1 / (1 + np.exp(-z))


def find_max_output(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    return index


def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)

