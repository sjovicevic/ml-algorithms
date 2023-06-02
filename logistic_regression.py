import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep


def compute_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_softmax(func_value, samples):
    return np.exp(func_value) / np.sum(np.exp(func_value), axis=1).reshape(samples, 1)


def transform_bias(a):
    ones_to_x = np.ones((a.shape[0], 1))
    return np.concatenate([ones_to_x, a], axis=1)


class LogisticRegression:

    def __init__(self, alpha=0.001, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.J_history = []
        self.binary = None

    def fit(self, x, y, binary=None):
        if binary:
            self.binary_classification(x, y)
        else:
            self.multiclass_classification(x, y)

    def binary_classification(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        X = transform_bias(X)
        for _ in range(self.n_iters):
            lin_prediction = np.dot(X, self.weights)
            predictions = compute_sigmoid(lin_prediction)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))

            self.weights -= self.alpha * dw

    def multiclass_classification(self, X, y):
        n_samples, n_features = X.shape
        y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
        n_classes = len(y_one_hot[0])

        self.weights = np.random.uniform(low=-1, high=1, size=(n_classes, n_features + 1))
        X = transform_bias(X)
        self.weights = self.weights.T
        loss = 0
        for _ in tqdm(range(self.n_iters), desc="Training progress: "):

            f_wb = np.dot(X, self.weights)
            softmax = compute_softmax(f_wb, n_samples)

            loss = (-1 / n_samples) * (np.sum(np.multiply(y_one_hot, np.log(softmax))))
            dw = (1 / n_samples) * np.dot(X.T, (y_one_hot - softmax))

            self.weights += self.alpha * dw
            self.J_history.append(loss)

            sleep(0.001)

        print(f'Loss value: [{loss}]')

    def predict(self, X):

        n_samples, n_features = X.shape
        X = transform_bias(X)
        f_wb = np.dot(X, self.weights)

        if self.binary:
            predictions = compute_sigmoid(f_wb)
            return predictions
        else:
            softmax = compute_softmax(f_wb, n_samples)
            indices = np.argmax(softmax, axis=1)
            return indices

    def plot_model(self):
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.grid(True)
        plt.plot(self.J_history, color='orange')
        plt.show()
