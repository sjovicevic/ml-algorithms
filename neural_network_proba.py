import numpy as np
import utils

a = np.array([[1, 5, 9],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4]])

print(a)
print(np.sum(a, axis=1, keepdims=True))

a = utils.softmax(a)
print(a)

print(np.sum(a, axis=1)[0])