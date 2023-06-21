import numpy as np

import utils


class Layer:

    def __init__(self, inputs, neurons=1):
        self.cache = {}
        self.weights = None
        self.bias = None
        self.neurons = neurons
        self.inputs = inputs
        self.initialize_weights(self.inputs.shape[1])

    def initialize_weights(self, n_features):
        self.weights = np.random.uniform(low=-1, high=1, size=(n_features, self.neurons))
        self.bias = np.zeros(self.neurons)

    def softmax(self):
        return np.exp(self.cache['Z']) / np.sum(np.exp(self.cache['Z']), axis=1, keepdims=True)

    def forward(self):
        z = np.dot(self.inputs, self.weights) + self.bias
        self.cache['Z'] = z
        activation = self.softmax()
        self.cache['Activation'] = activation
        print(f'Dot product value: {z}')
        print(f'Dot product shape: {z.shape}')
        print(self.cache['Activation'])


layer = Layer(np.array([[2, 3, 4],
                        [5, 6, 7]]), neurons=3)
layer.forward()


