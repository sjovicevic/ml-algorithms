import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils

seed(0)


def create(layers):
    """
    :param layers: Dictionary with 3 keys, input, output, activation_f
    :return: List of layers with initialized weights
    """
    return [Layer(_input=layer['input'], n_neurons=layer['output'], activation_f=layer['activation_f'])
            for layer in layers]


def set_lr(alpha, layers):
    """
    Method for setting learning rate for all layers.
    """
    for layer in layers:
        layer.learning_rate = alpha

    return layers


class Layer:

    def __init__(self,
                 _input,
                 n_neurons,
                 activation_f):

        self.input = _input
        self.n_neurons = n_neurons
        self.activation_f = activation_f
        self.memory = {}
        self.weights = np.random.uniform(low=-1, high=1, size=(self.input, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)
        self.learning_rate = 0.0001

    def forward(self, x):
        self.memory['Input'] = x
        self.memory['Z'] = np.dot(self.memory['Input'], self.weights)
        self.memory['Activation'] = self.activation_f(self.memory['Z'])
        return self.memory['Activation']

    def backward(self, previous_derivative, req_delta=False):
        da_dz = self.activation_f(self.memory['Z'], derivative=True)
        w_grad = np.dot(self.memory['Input'].T, previous_derivative * da_dz)
        b_grad = np.sum(da_dz * previous_derivative, axis=0)
        self.weights += -self.learning_rate * w_grad
        self.bias += -self.learning_rate * b_grad

        return np.dot(previous_derivative * da_dz, self.weights.T) if req_delta else None


class NeuralNetwork:

    def __init__(self, layers, epochs, learning_rate):
        self.validation_accuracy_history = []
        self.validation_loss_history = []
        self.training_accuracy_history = []
        self.training_loss_history = []
        self.epochs = epochs
        self.layers = set_lr(learning_rate, create(layers))

        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = None, None, None, None, None, \
            None

    def train(self, x_train, y_train, x_test, y_test, x_val, y_val):
        """
        'main' function of neural network.
        Cleaning layers, making prediction, calculating loss, getting loss derivative and then back propagating.
        Weights and biases are being updated on Layer level.
        Doing this 'epoch' times. You can set specific learning rate.
        """
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.X_val = x_val
        self.y_val = y_val

        for _ in tqdm(range(self.epochs), desc="Training progress: "):
            prediction = self.propagation(self.X_train)
            temp_loss = utils.loss(prediction, self.y_train)
            # training loss
            self.training_loss_history.append(temp_loss)
            loss_derivative = utils.loss_derivative(self.y_train, prediction)
            self.backpropagation(loss_derivative)

            # validation loss
            prediction = self.propagation(self.X_val)
            temp_loss = utils.loss(prediction, self.y_val)
            self.validation_loss_history.append(temp_loss)

            # validation accuracy
            self.validation_accuracy_history.append(self.calculate_accuracy(self.X_val, self.y_val))

            # training accuracy
            self.training_accuracy_history.append(self.calculate_accuracy(self.X_train, self.y_train))

    def propagation(self, x):
        """
        Forward propagation through every layer.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backpropagation(self, tmp_derivative):
        """
        Backpropagation through every layer.
        """
        for idx in range(len(self.layers) - 1, -1, -1):
            tmp_derivative = self.layers[idx].backward(tmp_derivative, idx != 0)

    def calculate_accuracy(self, x, y):
        y_predicted = np.argmax(self.propagation(x), axis=1)
        return utils.accuracy(y_predicted, y)

    def plot_loss(self):
        """
        Simple plotting.
        """
        plt.figure(figsize=(12, 7))

        plt.subplot(1, 2, 1)
        plt.plot(self.training_accuracy_history, c='orange', label='training_accuracy')
        plt.plot(self.validation_accuracy_history, c='blue', label='validation_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.training_loss_history, c='orange', label='training_loss')
        plt.plot(self.validation_loss_history, c='blue', label='validation_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=1234)

layers_in = [
            {'input': 4, 'output': 128, 'activation_f': utils.tanh},
            {'input': 128, 'output': 64, 'activation_f': utils.tanh},
            {'input': 64, 'output': 3, 'activation_f': utils.softmax}
            ]

nn = NeuralNetwork(layers_in, epochs=60, learning_rate=0.0001)
nn.train(X_train, Y_train, X_test, Y_test, X_val, Y_val)
print(f'Accuracy score: {nn.calculate_accuracy(X_test, Y_test)}')
nn.plot_loss()
