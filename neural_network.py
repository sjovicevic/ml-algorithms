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

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponents / np.sum(exponents, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:

    def forward(self, output, y_true):
        pass

    def calculate(self, output, y_true):
        samples_losses = self.forward(output, y_true)
        try:
            # noinspection PyTypeChecker
            data_loss = np.mean(samples_losses)
            return data_loss
        except ValueError:
            print("Union ndarray expected, got None instead.")


class LossCategoricalCrossEntropy(Loss):

    def forward(self, y_prediction, y_true):

        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        correct_confidence = numpy.ndarray
        if len(y_true.shape) == 1:
            correct_confidence = y_prediction_clipped[(range(samples), y_true)]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_prediction_clipped * y_true, axis=1)

        negative_log = -np.log(correct_confidence)
        return negative_log

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


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

predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy}')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Dark2')
plt.show()
