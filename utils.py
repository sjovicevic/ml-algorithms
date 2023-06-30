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


def softmax_derivative(z):
    return softmax(z) * (1 - softmax(z))


def categorical_cross_entropy_loss(y_hat, y=None, derivative=False):
    if derivative:
        return loss_derivative(y_hat, y)
    else:
        return -np.multiply(y, np.log(y_hat))


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


def loss(a, y, n_samples):
    y_one_hot = get_one_hot(y)
    return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(a)))


def find_max_output(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if output[i] > m:
            m, index = output[i], i
    return index


def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)


def plot_model(J_history):
    """
    Plotting the cost values.
    """
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.grid(True)
    plt.plot(J_history, color='orange')
    plt.show()
