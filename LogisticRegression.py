import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression():

    def __init__(self, alpha = 0.001, n_iters = 1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y, binary=True):
        if binary:
            self.binary_classification(X, y)
        else:
            self.multiclass_classification(X, y)



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
        n_classes = 3
        self.weights = np.random.uniform(low=-1, high=1, size=(n_classes, n_features))
        self.weights = self.weights.T
        self.bias = 0

        print(X.shape)
        for _ in range(self.n_iters):
            single_prediction = np.dot(X, self.weights) + self.bias
            print(f"Single prediction shape {single_prediction.shape}")
            single_prediction_exp = np.exp(single_prediction)
            print(f"Single prediction exponent shape {single_prediction_exp.shape}")
            scalar_sum = np.sum(np.exp(single_prediction), axis=1)
            print(f"Scalar sum value is {scalar_sum}, and shape {scalar_sum.shape}")
            print('-----------')
            scalar_sum_correct_shape = scalar_sum.reshape(120,1)
            print(f"Scalar sum shape now is {scalar_sum_correct_shape.shape}")

            predictions = single_prediction_exp / scalar_sum_correct_shape # moj kod

            print(f"Predictions shape {predictions.shape} and predictions values {predictions}")
            print(f"Y shape {y.shape}, and Y values {y}")

            y = y.reshape(120, 1)
            result = y - predictions

            print(f"Result shape is {result.shape}, and result value {result}")
            print(f"X shape : {X.shape}")

            dw = (1 / n_samples) * np.dot(result.T, X)
            db = (1 / n_samples) * np.sum(result)

            print(f"Weights shape: {self.weights.shape}")
            # print(f"Alpha shape: {self.alpha.shape}")
            print(f"Dw shape: { dw.shape }")

            dw = dw.T

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db





    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_predictions)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred