import numpy as np



def softmax(vector):
    result = np.exp(vector)
    return result / np.sum(np.exp(vector))

a = np.array([5,8,2,4,5])
vector = softmax(a)
for item in vector:
    print("%.3f" % (item))
