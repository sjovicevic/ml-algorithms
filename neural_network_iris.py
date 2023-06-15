import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import utils


class Layer:

    def __init__(self, input, n_neurons, output=False):
        self.input = input
        self.n_samples = self.input.shape[0]
        self.n_features = self.input.shape[1]
        self.n_neurons = n_neurons
        self.z = None
        self.output = output

        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_features, self.n_neurons))
        self.bias = np.zeros(self.n_neurons)

    def forward(self):
        self.z = np.dot(self.input, self.weights) + self.bias
        print(self.z)
        if self.output:
            self.z = utils.softmax(self.z)
        else:
            self.z = utils.sigmoid(self.z)
        print(self.z)


ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
print(f'X values: {X.shape}')
print(f'Y values: {y.shape}')

layer = Layer(X, 5, output=True)
layer.forward()