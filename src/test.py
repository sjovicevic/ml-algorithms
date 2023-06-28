import numpy as np

a = np.array([[1, 1, 1, 1],
              [24, 2, 1, 5],
              [9, 8, 3, 2],
              [9, 11, 23, 14],
              [34, 3, 1, 3]])

print(a)
print(np.diagflat(a))