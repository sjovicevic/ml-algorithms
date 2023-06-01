import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


iris = datasets.load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


clf = LogisticRegression(alpha=0.001, n_iters=300)
clf.fit(X_train, y_train, binary=False)
y_pred = clf.predict(X_test)
print(y_test)
acc = accuracy(y_pred, y_test)
print(f"Model accuracy: {acc}")
