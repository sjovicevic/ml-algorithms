import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, alpha=0.001, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.J_history = []

    def fit(self, x, y, binary=True):
        if binary:
            self.binary_classification(x, y)
        else:
            self.multiclass_classification(x, y)

    def binary_classification(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            lin_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(lin_prediction)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def multiclass_classification(self, X, y):
        n_samples, n_features = X.shape
        y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
        n_classes = len(y_one_hot[0])

        print(X)

        self.weights = np.random.uniform(low=-1, high=1, size=(n_classes, n_features)).T
        self.bias = 0

        print(f"X shape: {X.shape}")
        print(f"Y shape: {y.shape}")
        print(f"Y one hot shape: {y_one_hot.shape}")
        print(f"Weights shape: {self.weights.shape}")

        for _ in range(self.n_iters):
            f_wb = np.dot(X, self.weights) + self.bias

            softmax = np.exp(f_wb) / np.sum(np.exp(f_wb), axis=1).reshape(n_samples, 1)

            loss = (-1 / n_samples) * (np.sum(np.multiply(y_one_hot, np.log(softmax))))

            dw = (1 / n_samples) * np.dot(X.T, (y_one_hot - softmax))
            db = (1 / n_samples) * np.sum(y_one_hot - softmax)

            self.weights += self.alpha * dw
            self.bias += self.alpha * db

            self.J_history.append(loss)

            if _ % 100 == 0:
                print(f"Loss value {loss}")

        plt.plot(self.J_history, color='b')
        plt.show()

    def predict(self, X):
        n_samples, n_features = X.shape
        f_wb = np.dot(X, self.weights) + self.bias
        softmax = np.exp(f_wb) / np.sum(np.exp(f_wb), axis=1).reshape(n_samples, 1)
        indices = np.argmax(softmax, axis=1)
        print(indices)
        return indices
