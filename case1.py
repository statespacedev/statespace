
# linear bayesian processor, linear kalman filter

import numpy as np

def x(x, deltat, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
def y(x, v):
    return x**2 * x**3 + v
def A(x, deltat):
    return 1 - .05 * deltat + .08 * deltat * x
def C(x):
    return 2 * x + 3 * x**2

deltat = .01
tv = np.arange(0, 1.5, deltat)
n = t.shape[0]
vv = .09 * np.random.randn(n)

xf = 2.3
Pf = .01
Rww = 0
ref = .067 * t + 2.

pass