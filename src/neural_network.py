import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from tqdm import tqdm
import utils


class Layer:

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

    def forward(self, data, y=None):
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

            return delta.T


class NeuralNetwork:

    def __init__(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        self.X = x
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []

    def load(self, layer):
        self.layers.append(self.load_input_layer(layer['input']))
        self.layers.append(self.load_hidden_layer(layer['hidden']))
        self.layers.append(self.load_output_layer(layer['output']))

    def load_input_layer(self, layer_info: list):
        pass
        # return Layer(layer_info[0], layer_info[1])

    def load_hidden_layer(self, layer_info: list):
        pass

    def load_output_layer(self, layer_info: list):
        pass

    def train(self):
        pass


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

layers = {
    'input': [120, utils.relu],
    'hidden': [[20, utils.tanh], [10, utils.tanh]],
    'output': [3, utils.softmax]
}


for parameter, index in zip(params, range(len(params))):
    pass
