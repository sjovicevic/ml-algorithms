import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, copy

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))

    return g
def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    m, n = X.shape
    cost = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b

        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m

    reg_cost = 0

    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j] ** 2)
        reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost
def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]

        dj_db += err_i

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history


pd.set_option('display.max_columns', None)

iris = pd.read_csv('./data/iris.csv')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000
w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)

y_predicted = sigmoid(np.dot(X_train, w_out) + b_out)
print(w_out, type(w_out))
print(b_out, type(b_out))
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

