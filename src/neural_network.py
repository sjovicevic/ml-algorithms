import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import utils
'''
class Neuron:

    def __init__(self,
                 data,
                 n_neurons,
                 activation_f,
                 derivative_f,
                 output_neuron=False,
                 y=None):

        self.input = data
        self.n_neurons = n_neurons
        self.activation_f = activation_f
        self.derivative_f = derivative_f
        self.output_neuron = output_neuron
        self.y = y
        self.n_features = data.shape[1]
        self.memory = {}
        self.bias = 0
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))

        if y is not None and output_neuron:
            y_one_hot = utils.get_one_hot(y)
            if len(y_one_hot[0]) != n_neurons:
                raise Exception("Number of neurons must be equal to the number of the output classes.")

    def forward(self):
        self.memory['Z'] = np.dot(self.input, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        if self.output_neuron:
            loss = utils.loss(self.memory['Activation'], y_train, self.input.shape[0])
            return loss
        else:
            return self.memory['Activation']

    def backward(self, next_activation=None, next_weights=None, previous_derivative=None):

        if self.output_neuron:
            self.y = np.array(pd.get_dummies(self.y, dtype='int8'))
            self.memory['Loss_derivative_value'] = self.derivative_f(self.y, self.memory['Activation'])
            return self.memory['Loss_derivative_value']

        else:
            da_dz = self.derivative_f(self.memory['Z'])
            current_derivative = np.dot(next_activation.T, da_dz) * previous_derivative
            current_bias = np.sum(da_dz, axis=0, keepdims=True)
            delta = np.dot(next_weights.T, da_dz.T) * previous_derivative
            self.weights += -0.01 * current_derivative
            self.bias += -0.01 * current_bias

            return self.weights, delta.T, self.bias

'''


class NeuralNetwork:

    activation = {
        'hidden_layer_activation': utils.tanh,
        'output_layer_activation': utils.softmax
    }

    def __init__(self, X, y, layers: int, activation: dict):
        self.X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.y = y
        self.layers = layers
        self.activation = activation
        self.parameters = self.initialize()
        self.cache = {}
        self.input_layer = self.layers[0]
        self.hidden_layer = self.layers[1:-1:1]
        self.output_layer = self.layers[-1]

    def initialize(self):

        parameters = {}

        for layer, index in zip(self.layers, range(len(self.layers))):
            if index == 0:
                parameters[f'W{index}'] = np.random.uniform(low=-1, high=1, size=(self.n_features, layer))
                parameters[f'b{index}'] = 0
                previous_layer_number = layer
                continue

            parameters[f'W{index}'] = np.random.uniform(low=-1, high=1, size=(previous_layer_number, layer))
            parameters[f'b{index}'] = 0
            previous_layer_number = layer

        return parameters

    def optimizer(self):

        optimizers = {}

        for layer, index in zip(self.layers, range(len(self.layers))):
            optimizers[f'W{index}'] = np.zeros(self.parameters[f'W{index}'])
            optimizers[f'b{index}'] = np.zeros(self.parameters[f'b{index}'])

        return optimizers

    def forward(self):

        self.cache['X'] = self.X

        for index in range(len(self.hidden_layer)):
            if index == 0:
                self.cache[f'Z{index}'] = np.dot(self.parameters[f'W{index}'], self.cache[f'X'].T) + self.parameters[f'b{index}']
                self.cache[f'A{index}'] = self.activation['hidden_layer_activation'](self.cache[f'Z{index}'])
                continue
            self.cache[f'Z{index}'] = np.dot(self.parameters[f'W{index}'], self.cache[f'A{index-1}'].T) + self.parameters[f'b{index}']
            self.cache[f'A{index}'] = self.activation['hidden_layer_activation'](self.cache[f'Z{index}'])

        self.cache[f'Z{len(self.hidden_layer)}'] = np.dot(self.parameters[f'W{len(self.hidden_layer)}'], self.cache['X'])
        self.cache[f'A{len(self.hidden_layer)}'] = self.activation['output_layer_activation'](self.cache[f'Z{len(self.hidden_layer) - 1}'])

        return self.cache[f'A{len(self.hidden_layer)}']


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

nn = NeuralNetwork(X_train,
                   y_train,
                   [6, 4, 3],
                   {'hidden_layer_activation': utils.tanh, 'output_layer_activation': utils.softmax})

params = nn.initialize()
optims = nn.optimizer()
activation = nn.forward()

for parameter, index in zip(params, range(len(params))):
    pass
