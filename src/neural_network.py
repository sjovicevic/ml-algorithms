import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from tqdm import tqdm
import utils


class Layer:

    def __init__(self,
                 input,
                 n_neurons,
                 activation_f,
                 activation_f_derivative,
                 output_layer=None):
        self.input = input
        self.n_neurons = n_neurons
        self.activation_f = activation_f
        self.derivative_f = activation_f_derivative
        self.output_layer = output_layer
        self.y = y_train
        self.memory = {}
        self.bias = 0
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))

        if y is not None and output_layer:
            y_one_hot = utils.get_one_hot(y)
            if len(y_one_hot[0]) != n_neurons:
                raise Exception("Number of neurons must be equal to the number of the output classes.")

    def initialize_parameters(self, input):
        pass

    def forward(self):
        self.memory['Z'] = np.dot(self.input, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        if self.output_layer:
            loss = utils.loss(self.memory['Activation'], y_train, self.n_samples)
            return loss
        else:
            return self.memory['Activation']

    def backward(self, next_activation=None, next_weights=None, previous_derivative=None):

        if self.output_layer:
            self.y = np.array(pd.get_dummies(self.y, dtype='int8'))
            self.memory['Loss_derivative_value'] = self.derivative_f(self.memory['Activation'])
            #derivative = np.dot(self.memory['Loss_derivative_value'])
            return self.memory['Loss_derivative_value']

        else:
            da_dz = self.derivative_f(self.memory['Z'])
            current_derivative = np.dot(next_activation, da_dz) * previous_derivative
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

        """
        This function loads layers into neural network.
        :param layer: dictionary with keywords 'input', 'hidden', 'output' with list with two parameters as values.
        First parameter represents number of neurons, second parameter represents activation function.
        :return: List with Layer objects [Layer(input), Layer(hidden)..., Layer(output)]
        """
        _layers = [self.load_layer(layer['input'])]
        for layer_info in layer['hidden']:
            _layers.append(self.load_layer(layer_info))
        _layers.append(self.load_output_layer(layer['output']))

        self.layers = _layers
        return self.layers

    @staticmethod
    def load_layer(layer_info: list):
        return Layer(layer_info[0], layer_info[1])

    @staticmethod
    def load_output_layer(layer_info: list):
        return Layer(layer_info[0], layer_info[1], output_layer=True)

    def train(self):
        for _ in range(self.epochs):
            for layer, index in zip(self.layers, range(len(self.layers))):
                layer.forward()


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

layers = {
    'input': [120, utils.relu],
    'hidden': [[10, utils.tanh]],
    'output': [3, utils.softmax]
}
epochs = 10

input_layer = Layer(X_train,
                    120,
                    activation_f=utils.tanh,
                    activation_f_derivative=utils.tanh_derivative)
input_layer_output = input_layer.forward()
hidden_layer = Layer(input_layer_output,
                     10,
                     activation_f=utils.tanh,
                     activation_f_derivative=utils.tanh_derivative)
hidden_layer_output = hidden_layer.forward()
output_layer = Layer(hidden_layer_output,
                     3,
                     activation_f=utils.softmax,
                     activation_f_derivative=utils.softmax_derivative,
                     output_layer=True)
output_layer_output = output_layer.forward()

for _ in range(epochs):
    output_derivative = output_layer.backward(hidden_layer_output, hidden_layer.weights, output_layer_output)
    hidden_derivative = hidden_layer.backward(input_layer_output, input_layer.weights, output_derivative)
    input_derivative = input_layer.backward(input_layer_output, input_layer.weights, hidden_derivative)

    input_layer_output = input_layer.forward()
    hidden_layer_output = hidden_layer.forward()
    output_layer_output = output_layer.forward()

    print(output_layer_output)
