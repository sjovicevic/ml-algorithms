import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression
import utils

ldr = utils.DatasetLoader(dataset=datasets.load_iris(), multiclass_flag=True)
multiclass, X, y = ldr.run()
print(f'X values: {X}')
print(f'Y values: {y}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(alpha=0.01, n_iters=10)
clf.fit(X_train, y_train, multiclass)

y_prediction = clf.predict(X_test, multiclass)
print(f"Model prediction: {y_prediction}")
acc = utils.accuracy(y_prediction, y_test)
print(f"Model accuracy: {acc}")
clf.plot_model()
