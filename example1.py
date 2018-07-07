
# linear bayesian processor, linear kalman filter

import numpy as np

def A(x, deltat):
    return 1 - .05 * deltat + .08 * deltat * x
def C(x):
    return 2 * x + 3 * x**2

deltat = .01
t = np.arange(0, 1.5, deltat)
n = t.shape[0]
v = .09 * np.random.randn(n)
xf = 2.3
Pf = .01
Rww = 0
ref = .067 * t + 2.

pass