import numpy as np
import pandas as pd
import matplotlib as plt


class DatasetLoader:
    def __init__(self, dataset, multiclass_flag=False):
        self.dataset = dataset
        self.multiclass_flag = multiclass_flag

    def run(self):
        return self.multiclass_flag, self.dataset.data, self.dataset.target


def linear(z):
    return z


def softmax(z, derivative=False):
    if derivative:
        return softmax_derivative(z)
    else:
        e_z = np.exp(z - np.max(z, axis=1).reshape(-1,1))
        return e_z / np.sum(e_z, axis=1, keepdims=True)


def softmax_derivative(s):
    return softmax(s) * (1 - softmax(s))


'''
def softmax_derivative(dvalues, prediction):
    derivative = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(prediction, dvalues)):
        single_output = single_output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
        derivative[index] = np.dot(jacobian_matrix, single_dvalues)

    return derivative
'''

def categorical_cross_entropy_loss(y_hat, y, derivative=False):
    if derivative:
        return loss_derivative(y, y_hat)
    else:
        return -np.multiply(y, np.argmax(np.log(y_hat), axis=1))


def loss(a, y, derivative=False):
    if derivative:
        loss_derivative(y, a)
    else:
        n_samples = y.shape[0]
        y_one_hot = get_one_hot(y)
        return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(a)))


def loss_derivative(y, y_hat):
    y = get_one_hot(y)
    return -np.divide(y, y_hat)


def tanh(z, derivative=False):
    if derivative:
        return tanh_derivative(z)
    else:
        return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z)**2


def relu(z, derivative=False):
    if derivative:
        return relu_derivative(z)
    else:
        return np.maximum(0, z)


def relu_derivative(z):
    z = np.where(z < 0, 0, z)
    z = np.where(z >= 0, 1, z)
    return z


def sigmoid(z, derivative=False):
    if derivative:
        return sigmoid_derivative(z)
    else:
        return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return np.multiply(z, 1 - z)


def get_one_hot(z):
    return np.array(pd.get_dummies(z, dtype='int8'))


def loss(a, y):
    y_one_hot = get_one_hot(y)
    n_samples = a.shape[0]
    return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(a)))



def find_max_output(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if output[i] > m:
            m, index = output[i], i
    return index


def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)


def transform_bias(a):
    """
    Helper function that adds bias to features matrix.
    :param a: Feature matrix.
    :return: New matrix with one column shape + 1.
    """
    ones_to_x = np.ones((a.shape[0], 1))
    return np.concatenate([a, ones_to_x], axis=1)