import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import utils


def tanh(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)


def tanh_derivative(z):
    return 1 - tanh(z)**2


def loss(a, n_samples):
    y_one_hot = np.array(pd.get_dummies(y, dtype='int8'))
    return (-1 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(a)))


class Neuron:

    def __init__(self, n_features, activation_f, derivative_f, output_neuron=False):
        self.weights = np.random.uniform(low=-1, high=1, size=(n_features, 1))
        self.bias = 0
        self.activation_f = activation_f
        self.derivative_f = derivative_f
        self.memory = {}
        self.output_neuron = output_neuron

    def forward(self, inputs):
        self.memory['Z'] = np.dot(inputs, self.weights) + self.bias
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        if self.output_neuron:
            loss = utils.loss(self.memory['Activation'], y_train, inputs.shape[0])
            print(loss)
        return self.memory['Activation']

    def backward(self, next_activation, previous_derivative):
        da_dz = self.derivative_f(self.memory['Z'])
        current_derivative = np.multiply(previous_derivative, da_dz)
        self.weights += 0.01 * current_derivative
        return self.weights


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

neuron1 = Neuron(X_train.shape[1], tanh, tanh_derivative, output_neuron=True)
neuron1_output = neuron1.forward(X_train)
print(neuron1.backward(1,1))