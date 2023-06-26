import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import utils


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
        print(f'Z shape: {self.memory["Z"].shape}')
        self.memory['Activation'] = self.activation_f(self.memory['Z'])

        # if self.output_neuron:
        #     loss = utils.loss(self.memory['Activation'], y_train, inputs.shape[0])
        #     print(loss)

        return self.memory['Activation']

    def backward(self, next_activation, next_weights, previous_derivative):
        da_dz = self.derivative_f(self.memory['Z'])
        print(f'Shape of next_activation: {next_activation.shape}')
        print(f'Shape of da_dz: {da_dz.shape}')
        current_derivative = np.dot(next_activation.T, da_dz) * previous_derivative
        delta = np.dot(next_weights.T, da_dz.T) * previous_derivative
        self.weights += 0.01 * current_derivative
        return self.weights, delta.T


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

a = np.array([[0.1, 0.3, 0.1, 0.13],
              [0.45, 2.34, 1.2, 5.5],
              [0.87, 2.35, 2.21, 8.9]])

neuron1 = Neuron(a.shape[1], utils.tanh, utils.tanh_derivative, output_neuron=False)
neuron1_output = neuron1.forward(a)
weights, delta = neuron1.backward(neuron1_output, np.array([[0.01, 0.02, 0.03, 0.04]]), 1)


print(f'Weights value after: {weights}')
print(f'Delta: {delta}')
