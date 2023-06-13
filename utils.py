import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def relu(z):
    return np.maximum(0, z)