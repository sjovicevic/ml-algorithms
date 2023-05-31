import numpy as np
import pandas as pd

y = np.array([[1],[1],[1],[1]])
predictions = np.array([[1, 2], [2, 1], [1, 1], [1, 1]])
x = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
print(predictions.shape)
print(y)
print(y.shape)
tr =  y - predictions
print(tr)
print(tr.shape)

print('-----------')
print(np.dot(tr.T, x).shape)