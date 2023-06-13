import numpy as np
import matplotlib.pyplot as plt
import utils


class HiddenLayer:

    def __init__(self, input, n_neurons):
        self.input = input
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.n_neurons = n_neurons
        self.z = None
        self.weights = np.random.rand(self.n_features, self.n_neurons)
        self.bias = np.zeros(self.n_neurons)

    def forward(self):

        z = np.dot(self.input, self.weights) + self.bias
        print(f'z = {z}')
        z = utils.softmax(z)
        print(f'z = {z}')

    def back(self):
        pass


X = np.array([[1, 2, 3],
              [4, 1, 2],
              [4, 1, 1],
              [3, 9, 0]])

layer = HiddenLayer(X, 3)
layer.forward()
