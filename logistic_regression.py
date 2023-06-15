import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import math


def compute_sigmoid(f_wb):
    """
    Sigmoid function, mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    Function that maps input values to a value between 0 and 1.
    :param f_wb: function f_wb result
    :return: float between 0 and 1
    """
    return 1 / (1 + np.exp(-f_wb))


def compute_softmax(func_value):
    """
    Softmax function converts a vector of K real numbers into a probability distribution of K possible outcomes.
    :return: A matrix with shape (n_samples, n_classes) that represents softmax value for every element in every class.
    """
    return np.exp(func_value) / np.sum(np.exp(func_value), axis=1, keepdims=True)


def transform_bias(a):
    """
    Helper function that adds bias to features matrix.
    :param a: Feature matrix.
    :return: New matrix with one column shape + 1.
    """
    ones_to_x = np.ones((a.shape[0], 1))
    return np.concatenate([ones_to_x, a], axis=1)


class LogisticRegression:

    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.J_history = []
        self.loss = None
        self.y_one_hot = None

    def fit(self, X, y, multiclass=True):
        """
        Fit function that calculates the weights and bias for the model.
        :param X: X_train [n_samples x n_features]
        :param y: y_train [n_samples x 1]
        :param multiclass: Flag parameter to decide which type of regression we want.
        :return: Nothing, it updates class property [self.weights]
        """
        n_samples, n_features = X.shape

        if not multiclass:
            self.weights = np.zeros(n_features + 1)
        else:
            self.y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
            n_classes = len(self.y_one_hot[0])
            self.weights = np.random.uniform(low=-1, high=1, size=(n_features + 1, n_classes))

        X = transform_bias(X)

        print(f'X shape: {X.shape}')
        print(f'Y shape: {y.shape}')
        print(f'weights shape: {self.weights.shape}')
        for _ in tqdm(range(self.n_iters), desc="Training progress: "):
            f_wb = np.dot(X, self.weights)

            if not multiclass:
                dw = self.binary_classification(X, y, f_wb, n_samples)
            else:
                dw = self.multiclass_classification(X, f_wb, n_samples)

            self.weights -= self.alpha * dw

        print(f'Loss value: [{self.loss}]')

    def binary_classification(self, X, y, f_wb, n_samples):
        """
        Modular implementation of logistic regression with binary output.
        :param X: X_train
        :param y: y_train
        :param f_wb: Function f that depends on parameters w, b, and X
        :param n_samples: Number of samples in the dataset.
        :return: Derivative of weights.
        """
        predictions = compute_sigmoid(f_wb)
        self.loss = (-1 / n_samples) * (np.dot(y.T, predictions) + np.dot((1 - y).T, np.log(1 - predictions)))
        self.J_history.append(self.loss)
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))

        return dw

    def multiclass_classification(self, X, f_wb, n_samples):
        """
        Modular implementation of logistic regression with multiclass output.
        :param X: X_train
        :param y: y_train
        :param f_wb: Function f that depends on the parameters w, b, and X
        :param n_samples: Number of samples in the dataset.
        :return:
        """
        softmax = compute_softmax(f_wb)
        print(f"softmax{softmax}")
        self.loss = (-1 / n_samples) * np.sum(np.multiply(self.y_one_hot, np.log(softmax)))
        self.J_history.append(self.loss)
        dw = (-1 / n_samples) * np.dot(X.T, (self.y_one_hot - softmax))

        return dw

    def predict(self, X, multiclass=True):
        """
        Calculate the accuracy on testing set.
        :param X: Testing set.
        :param multiclass: Flag whether output is binary or multiclass problem.
        :return: Indices of predicted class inside one row for multiclass case. Or predicted value for binary case.
        """
        n_samples, n_features = X.shape
        X = transform_bias(X)
        f_wb = np.dot(X, self.weights)

        if not multiclass:
            predictions = compute_sigmoid(f_wb)
            return predictions
        else:
            softmax = compute_softmax(f_wb)
            indices = np.argmax(softmax, axis=1)
            return indices

    def plot_model(self):
        """
        Plotting the cost values.
        """
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.grid(True)
        plt.plot(self.J_history, color='orange')
        plt.show()
