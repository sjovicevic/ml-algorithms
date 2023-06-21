import numpy as np


class DatasetLoader:
    def __init__(self, dataset, multiclass_flag=False):
        self.dataset = dataset
        self.multiclass_flag = multiclass_flag

    def run(self):
        return self.multiclass_flag, self.dataset.data, self.dataset.target


def softmax(z, derivative=False):
    if derivative:
        return np.diagflat(z) - np.dot(z, z.T)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def cross_entropy_loss(y_one_hot, n_samples, softmax, derivative=False):
    if derivative:
        pass
        # derivative implementation
    return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(softmax)))


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

