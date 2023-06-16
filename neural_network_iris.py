import numpy as np
import pandas as pd
from sklearn import datasets
import utils


class Layer:

    def __init__(self, input, n_neurons, activation_f):
        self.input = input
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.n_neurons = n_neurons
        self.z = None
        self.cache = {}
        self.activation_f = activation_f
        self.grads = None

        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)

    def forward(self):
        self.cache['Z'] = np.dot(self.input, self.weights) + self.bias

        self.cache['Activation'] = self.activation_f(self.cache['Z'])

        return self.cache['Activation']

    def compute_loss(self, output):
        pass

    def back(self, expected, output):
        output_one_hot = np.array(pd.get_dummies(output, dtype='int8'))

        dz = output - expected.T
        self.weights = (1 / self.n_samples) * np.matmul(dz, self.cache['Activation'].T)
        db = (1 / self.n_samples) * np.sum(dz, axis=1, keepdims=True)


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
print(f'X values: {X.shape}')
print(f'Y values: {y.shape}')

layer = Layer(X, 3, activation_f=utils.softmax)
layer.forward()