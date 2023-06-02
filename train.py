import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression


def accuracy(y_p, y_t):
    return np.sum(y_p == y_t) / len(y_t)


iris = datasets.load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(alpha=0.01, n_iters=1000)
clf.fit(X_train, y_train, binary=False)
y_prediction = clf.predict(X_test)
acc = accuracy(y_prediction, y_test)
print(f"Model accuracy: {acc}")
