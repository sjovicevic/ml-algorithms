import numpy as np
import pandas as pd

arr = np.array([[0,1,0],
        [1,0,0],
       [1,0,0],
       [0,0,1]])

max_indices = np.argmax(arr, axis=1)

# Create a new matrix with the same shape as input_matrix
mapped_matrix = np.zeros_like(arr)

# Set the value at the max index to 1, others to 0
mapped_matrix[np.arange(arr.shape[0]), max_indices] = 1

print(mapped_matrix)