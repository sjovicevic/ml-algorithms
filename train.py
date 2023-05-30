import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

bc = datasets.load_breast_cancer()
iris = datasets.load_iris()
X, y = bc.data, bc.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(alpha=0.1)
clf.fit(X_test, y_test)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred, y_test)
print(acc)