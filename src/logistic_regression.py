import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils





class LogisticRegression:

    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.J_history = []
        self.loss = None
        self.y_one_hot = None

    def fit(self, x, y, multiclass=True):
        """
        Fit function that calculates the weights and bias for the model.
        :param x: X_train [n_samples x n_features]
        :param y: y_train [n_samples x 1]
        :param multiclass: Flag parameter to decide which type of regression we want.
        :return: Nothing, it updates class property [self.weights]
        """
        n_samples, n_features = x.shape

        if not multiclass:
            self.weights = np.zeros(n_features + 1)
        else:
            self.y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
            n_classes = len(self.y_one_hot[0])
            self.weights = np.random.uniform(low=-1, high=1, size=(n_features + 1, n_classes))

        x = utils.transform_bias(x)

        for _ in tqdm(range(self.n_iters), desc="Training progress: "):
            f_wb = np.dot(x, self.weights)

            if not multiclass:
                dw = self.binary_classification(x, y, f_wb, n_samples)
            else:
                dw = self.multiclass_classification(x, f_wb, n_samples)

            self.weights -= self.alpha * dw

        print(f'Loss value: [{self.loss}]')

    def binary_classification(self, x, y, f_wb, n_samples):
        """
        Modular implementation of logistic regression with binary output.
        :param x: X_train
        :param y: y_train
        :param f_wb: Function f that depends on parameters w, b, and X
        :param n_samples: Number of samples in the dataset.
        :return: Derivative of weights.
        """
        predictions = utils.sigmoid(f_wb)
        self.loss = (-1 / n_samples) * (np.dot(y.T, predictions) + np.dot((1 - y).T, np.log(1 - predictions)))
        self.J_history.append(self.loss)
        dw = (1 / n_samples) * np.dot(x.T, (predictions - y))

        return dw

    def multiclass_classification(self, x, f_wb, n_samples):
        """
        Modular implementation of logistic regression with multiclass output.
        :param x: X_train
        :param f_wb: Function f that depends on the parameters w, b, and X
        :param n_samples: Number of samples in the dataset.
        :return:
        """
        softmax = utils.softmax(f_wb)
        self.loss = (-1 / n_samples) * np.sum(np.multiply(self.y_one_hot, np.log(softmax)))
        self.J_history.append(self.loss)
        dw = (-1 / n_samples) * np.dot(x.T, (self.y_one_hot - softmax))

        return dw

    def predict(self, x, multiclass=True):
        """
        Calculate the accuracy on testing set.
        :param x: Testing set.
        :param multiclass: Flag whether output is binary or multiclass problem.
        :return: Indices of predicted class inside one row for multiclass case. Or predicted value for binary case.
        """
        x = utils.transform_bias(x)
        f_wb = np.dot(x, self.weights)

        if not multiclass:
            predictions = utils.sigmoid(f_wb)
            return predictions
        else:
            softmax = utils.softmax(f_wb)
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
