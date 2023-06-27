import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import utils
from tqdm import tqdm


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

    def initialize(self):

        input_layer = self.layers[0]
        hidden_layer = self.layers[1:-1:1]
        output_layer = self.layers[-1]

        parameters = {}

        for layer, index in zip(self.layers, range(len(self.layers))):
            if index == 0:
                parameters[f'W{index}'] = np.random.uniform(low=-1, high=1, size=(self.n_features, layer))
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



ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

a = np.array([[0.1, 0.3, 0.1, 0.13],
              [0.45, 2.34, 1.2, 5.5],
              [0.87, 2.35, 2.21, 8.9]])


neuron1 = Neuron(X_train, 12, utils.tanh, utils.tanh_derivative, output_neuron=False)
neuron2 = Neuron(neuron1.forward(), 6, utils.tanh, utils.tanh_derivative, output_neuron=False, y=None)
neuron3 = Neuron(neuron2.forward(), 3, utils.softmax, utils.loss_derivative, output_neuron=True, y=y_train)

nn = [neuron1, neuron2, neuron3]


neural_n = NeuralNetwork(1, 0.01, nn)
neural_n.train()



