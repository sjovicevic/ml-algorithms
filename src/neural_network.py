import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import utils


class Layer:

    def __init__(self,
                 input,
                 n_neurons,
                 activation_f,
                 activation_f_derivative,):
        self.input = input
        self.n_neurons = n_neurons
        self.activation_f = activation_f
        self.derivative_f = activation_f_derivative
        self.y = y_train
        self.memory = {}
        self.n_features = input.shape[1]
        self.n_samples = input.shape[0]
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)

    def __add__(self, other: dict):
        return Layer(
            self.forward(self.input),
            other['neurons'],
            other['activation_f'],
            other['activation_f_derivative'])

    def init_parameters(self, input):
        pass

    def forward(self, x):
        self.input = x
        self.memory['Z'] = np.dot(self.input, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])
        return self.memory['Activation']

    def backward(self, previous_derivative=None):
        da_dz = self.derivative_f(self.memory['Z'])
        delta = np.dot(previous_derivative * da_dz, self.weights.T)
        w_grad = np.dot(self.input.T, previous_derivative * da_dz)
        # b_grad = np.sum(previous_derivative * da_dz, axis=0)

        self.weights += -0.001 * w_grad
        # self.bias += -0.001 * b_grad

        return delta


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
        _layers.append(self.load_layer(layer['output']))

        self.layers = _layers

        return self.layers

    def train(self, x, y):
        input_layer_out = None
        output_layer_out = None
        for index, layer in enumerate(self.layers):
            if index == 0:
                layer.init_parameters(x)
                input_layer_out = layer.forward(x)
                continue
            elif index != len(self.layers) - 1:
                next_layer = self.layers[index + 1]
                next_layer.init_parameters(input_layer_out)
                input_layer_out = next_layer.forward(input_layer_out)
            else:
                layer.init_parameters(input_layer_out)
                output_layer_out = layer.forward(input_layer_out)
        for layer in self.layers:
            print(layer.weights.shape)


    @staticmethod
    def load_layer(layer_info: list, _input=False):
        return Layer(layer_info[0], layer_info[1], layer_info[2])


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

epochs = 20

l1 = Layer(X_train, 30, utils.tanh, utils.tanh_derivative)
l2 = l1.__add__({'neurons': 10, 'activation_f': utils.tanh, 'activation_f_derivative': utils.tanh_derivative})
l3 = l2.__add__({'neurons': 3, 'activation_f': utils.softmax, 'activation_f_derivative': utils.softmax_derivative})

loss_history = []

for _ in range(epochs):
    y_hat = l3.forward(l2.forward(l1.forward(X_train)))
    loss = utils.loss(y_hat, y_train, X_train.shape[0])
    loss_history.append(loss)

    loss_derivative = utils.loss_derivative(y_train, y_hat)
    lay3_back = l3.backward(loss_derivative)
    lay2_back = l2.backward(lay3_back)
    lay1_back = l1.backward(lay2_back)

plt.plot(loss_history, c='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
