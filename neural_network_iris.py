import numpy as np
import pandas as pd
from sklearn import datasets
import utils


class Layer:

    def __init__(self, input, n_neurons, output=False):
        self.input = input
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.n_neurons = n_neurons
        self.z = None
        self.output = output
        self.cache = {}

        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)

    def forward(self):
        self.cache['Z'] = np.dot(self.input, self.weights) + self.bias

        if self.output:
            self.cache['Activation'] = utils.softmax(self.cache['Z'])
        else:
            self.cache['Activation'] = utils.sigmoid(self.cache['Z'])

        return self.cache['Activation']

    def compute_loss(self, output):
        if self.output:
            pass
        else:
            raise Exception("Loss can be calculated only for output layer.")

    def back(self, expected, output):
        self.output_one_hot = np.array(pd.get_dummies(output, dtype='int8'))


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
print(f'X values: {X.shape}')
print(f'Y values: {y.shape}')

layer = Layer(X, 3, output=True)
layer.forward()