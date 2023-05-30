import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression():

    def __init__(self, alpha = 0.001, n_iters = 1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if len(set(y)) > 1:
            self.log_loss_descent(X, y, n_samples, n_features)
        else:
            pass



    def log_loss_descent(self, X, y, n_samples, n_features):
        for _ in range(self.n_iters):
            lin_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(lin_prediction)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def softmax(self, vector):
        result = np.exp(vector)
        return result / np.sum(np.exp(vector))

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_predictions)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred