import numpy as np
import pandas as pd
from sklearn import datasets
import utils


class Layer:

    def __init__(self, input, output, n_neurons, activation_f):
        self.input = input
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.n_neurons = n_neurons
        self.z = None
        self.cache = {}
        self.activation_f = activation_f
        self.grads = None
        self.output = output
        self.weights = None
        self.y_one_hot = None
        self.expected_value = None

        #self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)

    def initialize(self):
        self.y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
        n_classes = len(self.y_one_hot[0])
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, n_classes))

    def forward(self):
        self.cache['Z'] = np.dot(self.input, self.weights) + self.bias

        self.cache['Activation'] = self.activation_f(self.cache['Z'])

        return self.cache['Activation']

    def compute_loss(self, output):
        pass

    def back(self, expected, output):
        dz = output - expected.T
        self.weights = (1 / self.n_samples) * np.matmul(dz, self.cache['Activation'].T)
        self.bias = (1 / self.n_samples) * np.sum(dz, axis=1, keepdims=True)


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
print(f'X values: {X.shape}')
print(f'Y values: {y.shape}')

layer = Layer(X, y, 1, activation_f=utils.softmax)
layer.initialize()
print(layer.forward())