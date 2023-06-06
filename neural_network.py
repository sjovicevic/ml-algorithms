import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import compute_softmax
from sklearn import datasets
from sklearn.model_selection import train_test_split


class NeuralNetwork:

    def __init__(self, alpha=0.01, n_iters=1000, n_layers=2):
        self.alpha = alpha
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.weights = None
        self.bias = None
        self.J_history = []

    def fit(self, input_layer, output_layer):
        """
        Base function for fitting the neural network.
        :param input_layer: previously represented as X
        :param output_layer: previously represented as y
        :return: nothing, updates the parameters
        """
        print(f'Input layer shape: {input_layer.shape}')
        print(f'Output layer shape: {output_layer.shape}')


nn = NeuralNetwork(alpha=0.01, n_iters=1000, n_layers=2)
data = datasets.load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
nn.fit(X_train, y_train)
