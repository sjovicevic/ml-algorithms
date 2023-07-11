import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils


def create(layers):
    '''

    :param layers: Dictionary with 3 keys, input, output, activation_f
    :return: List of layers with initialized weights
    '''
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
        self.weights = np.random.uniform(low=-1, high=1, size=(self.input + 1, self.n_neurons))
        # self.bias = np.zeros(self.n_neurons)
        self.learning_rate = 0.01

    def forward(self, x):
        self.memory['Input'] = x
        self.memory['Z'] = np.dot(self.memory['Input'], self.weights)
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        # ovde prosirujem za plus 1
        return self.memory['Activation']

    def backward(self, previous_derivative, req_delta=False):
        da_dz = self.activation_f(self.memory['Z'], derivative=True)
        w_grad = np.dot(self.memory['Input'].T, previous_derivative * da_dz)
        # b_grad = np.sum(da_dz, axis=0)
        self.weights += -self.learning_rate * w_grad
        # self.bias += -self.learning_rate * b_grad

        return np.dot(previous_derivative * da_dz, self.weights.T) if req_delta else None


class NeuralNetwork:

    def __init__(self, layers, epochs, learning_rate):
        self.loss_history = []
        self.epochs = epochs
        self.layers = set_lr(learning_rate, create(layers))

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self, x_train, y_train, x_test, y_test):
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

        for _ in tqdm(range(self.epochs), desc="Training progress: "):
            prediction = self.propagation(self.X_train)
            temp_loss = utils.loss(prediction, y_train, self.X_train.shape[0])
            self.loss_history.append(temp_loss)
            loss_derivative = utils.loss_derivative(y_train, prediction)
            self.backpropagation(loss_derivative)

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

    def predict(self):
        y_predicted = np.argmax(self.propagation(self.X_test), axis=1)
        print(f'Predicted values: {y_predicted}')
        print(f'Actual values:    {self.y_test}')
        return utils.accuracy(y_predicted, self.y_test)

    def plot_loss(self):
        """
        Simple plotting.
        """
        #for loss in self.loss_history[0:-1:20]:
       #     print(f'Loss value: {loss:.5f}')

        plt.plot(self.loss_history, c='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train = utils.transform_bias(X_train)

layers_in = [
            {'input': X_train.shape[1], 'output': 16, 'activation_f': utils.tanh},
            {'input': 16, 'output': 8, 'activation_f': utils.tanh},
            {'input': 8, 'output': 3, 'activation_f': utils.softmax}
            ]

nn = NeuralNetwork(layers_in, epochs=100, learning_rate=0.0001)
nn.train(X_train, Y_train, X_test, Y_test)
print(f'Accuracy score: {nn.predict()}')
nn.plot_loss()

