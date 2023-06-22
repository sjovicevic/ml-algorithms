import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import utils

import utils


class Layer:

    def __init__(self, inputs, activation, neurons=1):
        self.cache = {}
        self.weights = None
        self.bias = None
        self.neurons = neurons
        self.inputs = inputs
        self.initialize_weights(self.inputs.shape[1])

        if activation == 'softmax':
            self.activation = self.softmax
        elif activation == 'relu':
            self.activation = self.relu
        elif activation == 'tanh':
            self.activation = self.tanh

    def initialize_weights(self, n_features):
        self.weights = np.random.uniform(low=-1, high=1, size=(n_features, self.neurons))
        self.bias = np.zeros(self.neurons)

    def softmax(self, derivative=False):
        if derivative:
            pass
        else:
            return np.exp(self.cache['Z']) / np.sum(np.exp(self.cache['Z']), axis=1, keepdims=True)

    def argmax(self):
        return np.argmax(self.cache['Activation'], axis=1, keepdims=True)

    def loss(self, A, outputs, derivative=False):
        y_one_hot = np.array(pd.get_dummies(outputs, dtype='int8'))

        if derivative:
            return np.divide(-y_one_hot, A)
        else:
            n_samples = self.inputs.shape[0]
            return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(A)))

    def tanh(self, derivative=False):
        if derivative:
            pass
        else:
            pass

    def forward(self, outputs):
        self.cache['Z'] = np.dot(self.inputs, self.weights) + self.bias
        self.cache['Activation'] = self.activation()

        # will leave this for last layer
        # loss = self.loss(self.cache['Activation'], outputs)
        print(self.cache['Activation'])
        return self.cache['Activation']

    def backward(self):
        pass


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

layer = Layer(X_train, 'softmax', 3)
layer.forward(y_train)
