import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponents / np.sum(exponents, axis=1, keepdims=True)


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
dense1.forward(X)
activation1 = ActivationReLU()
activation1.forward(dense1.output)
print(dense1.output[:5])
print(activation1.output[:5])
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2')
plt.show()

