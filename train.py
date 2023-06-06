import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork


class DatasetLoader:

    def __init__(self, dataset, multiclass_flag=False):
        self.dataset = dataset
        self.multiclass_flag = multiclass_flag

    def run(self):
        return self.multiclass_flag, self.dataset.data, self.dataset.target


def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)


ldr = DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
#ldr = DatasetLoader(dataset=datasets.load_breast_cancer(), multiclass_flag=False)
multiclass, X, y = ldr.run()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(alpha=0.01, n_iters=1000)
clf.fit(X_train, y_train, multiclass)

nn = NeuralNetwork(alpha=0.01, n_iters=1000, n_layers=2)
y_prediction = clf.predict(X_test, multiclass)
print(f"Model prediction: {y_prediction}")
acc = accuracy(y_prediction, y_test)
print(f"Model accuracy: {acc}")
clf.plot_model()
