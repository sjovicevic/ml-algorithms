import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        self.memory = {}
        self.n_features = input.shape[1]
        self.n_samples = input.shape[0]
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)
        self.learning_rate = 0

    def __add__(self, other: dict):
        return Layer(
            self.forward(self.input),
            other['neurons'],
            other['activation_f'],
            other['activation_f_d'])

    def forward(self, x):
        self.input = x
        self.memory['Z'] = np.dot(self.input, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        return self.memory['Activation']

    def backward(self, previous_derivative=None):
        da_dz = self.derivative_f(self.memory['Z'])
        delta = np.dot(previous_derivative * da_dz, self.weights.T)
        w_grad = np.dot(self.input.T, previous_derivative * da_dz)
        b_grad = np.sum(previous_derivative * da_dz, axis=0)
        self.weights += -self.learning_rate * w_grad
        self.bias += -self.learning_rate * b_grad

        return delta


class NeuralNetwork:

    def __init__(self, layers, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss_history = []
        self.layers = self.clean(layers)

    def train(self, epochs, learning_rate):
        """
        'main' function in neural network.
        Cleaning layers, making prediction, calculating loss, getting loss derivative and then back propagating.
        Weights and biases are updating on Layer level.
        Doing this 'epoch' times, also can set specific learning rate.
        """
        self.set_lr(learning_rate, self.layers)

        for _ in tqdm(range(epochs), desc="Training progress: "):
            prediction = self.propagation()
            temp_loss = utils.loss(prediction, self.y_train, self.X_train.shape[0])
            self.loss_history.append(temp_loss)
            loss_derivative = utils.loss_derivative(self.y_train, prediction)
            self.backpropagation(loss_derivative)

    @staticmethod
    def set_lr(alpha, layers):
        """
        Static method for setting learning rate for all layers.
        """
        for layer in layers:
            layer.learning_rate = alpha

        return layers

    def backpropagation(self, loss_der):
        """
        Backpropagation through every layer, not dependent on network size.
        """
        layer_in = self.layers[::-1]
        tmp_derivative = layer_in[0].backward(loss_der)
        for layer, index in zip(layer_in, range(len(layer_in) - 1)):
            tmp_derivative = layer_in[index + 1].backward(tmp_derivative)

    def propagation(self):
        """
        Forward propagation through every layer, not dependent on network size.
        """
        tmp_hat = self.layers[0]
        for layer, index in zip(layers_in, range(len(self.layers) - 1)):
            tmp_hat = self.layers[index + 1].forward(self.layers[index].memory['Activation'])

        return tmp_hat

    def clean(self, layers):
        """
        Method used to help set layers, extracting data from dictionary to a format suitable for use.
        """
        l1 = Layer(self.X_train, layers[0]['neurons'], layers[0]['activation_f'], layers[0]['activation_f_d'])
        tmp = l1
        index_layer = 1
        layers_out = [l1]

        for index in range(len(layers) - 1):
            layers_out.append(tmp.__add__(layers[index_layer]))
            tmp = layers_out[-1]
            index_layer += 1

        return layers_out

    def predict(self):
        pass

    def plot_loss(self):
        """
        Simple plotting.
        """
        plt.plot(self.loss_history, c='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

layers_in = [
            {'neurons': 30, 'activation_f': utils.tanh, 'activation_f_d': utils.tanh_derivative},
            {'neurons': 10, 'activation_f': utils.tanh, 'activation_f_d': utils.tanh_derivative},
            {'neurons': 3, 'activation_f': utils.softmax, 'activation_f_d': utils.softmax_derivative}
            ]

nn = NeuralNetwork(layers_in, X_train, X_test, Y_train, Y_test)
nn.train(epochs=30, learning_rate=0.001)
nn.plot_loss()
nn.predict()
