import numpy
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


class Loss:

    def forward(self, output, y_true):
        pass

    def calculate(self, output, y_true):
        samples_losses = self.forward(output, y_true)
        data_loss = np.mean(samples_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidence = numpy.ndarray
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[(range(samples), y_true)]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log = -np.log(correct_confidence)
        return negative_log


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
dense1.forward(X)
activation1 = ActivationReLU()
activation1.forward(dense1.output)
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()
loss_function = LossCategoricalCrossEntropy()
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)
print(loss)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2')
#plt.show()
