import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from logistic_regression import compute_sigmoid

nnfs.init()

class ActivationSoftmax:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_val / np.sum(exp_val, axis=1, keepdims=True)

class ActivationReLU:

    def __init__(self):
        self.output = 0

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
dense1.forward(X)
activation1 = ActivationReLU()
activation1.forward(dense1.output)
