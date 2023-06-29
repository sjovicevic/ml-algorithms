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
                 activation_f_derivative,
                 input_layer=None):
        self.input = input
        self.n_neurons = n_neurons
        self.activation_f = activation_f
        self.derivative_f = activation_f_derivative
        self.input_layer = input_layer
        self.y = y_train
        self.memory = {}
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]

        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(n_neurons)

    def forward(self):
        self.memory['Z'] = np.dot(self.input, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])
        return self.memory['Activation']

    def backward(self, next_activation=None, previous_derivative=None):
        da_dz = self.derivative_f(self.memory['Z'])
        delta = np.dot(previous_derivative * da_dz, self.weights.T)
        w_grad = np.dot(next_activation.T, previous_derivative * da_dz)
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
epochs = 80

lay1 = Layer(X_train, 30, utils.tanh, utils.tanh_derivative, input_layer=True)
lay1_out = lay1.forward()
lay2 = Layer(lay1_out, 10, utils.tanh, utils.tanh_derivative)
lay2_out = lay2.forward()
lay3 = Layer(lay2_out, 3, utils.softmax, utils.softmax_derivative)
lay3_out = lay3.forward()

loss_history = []

for _ in range(epochs):

    loss = utils.loss(lay3_out, y_train, X_train.shape[0])
    loss_history.append(loss)

    loss_derivative = utils.loss_derivative(y_train, lay3_out)

    lay3_back = lay3.backward(lay2_out, loss_derivative)
    lay2_back = lay2.backward(lay1_out, lay3_back)
    lay1_back = lay1.backward(X_train, lay2_back)

    lay1_out = lay1.forward()
    lay2_out = lay2.forward()
    lay3_out = lay3.forward()

plt.plot(loss_history, c='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()