import numpy as np

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0


xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
izlaz = xw0 + xw1 + xw2 + b
print(izlaz)

y = max(izlaz, 0)

dvalue = 1.0

drelu_dz = dvalue * (1. if izlaz > 0 else 0.)

dsum_dwx0 = 1
drelu_dwx0 = dsum_dwx0 * drelu_dz

dsum_dwx1 = 1
drelu_dwx1 = dsum_dwx1 * drelu_dz

dsum_dwx2 = 1
drelu_dwx2 = dsum_dwx2 * drelu_dz

print(drelu_dwx0, drelu_dwx1, drelu_dwx2)
print(drelu_dz)