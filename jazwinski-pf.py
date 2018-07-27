# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import util, math, plots

nsamp = 250
n = 150
deltat = .01


tts = np.arange(0, n * deltat, deltat)
Rww = 1e-6
wts = math.sqrt(Rww) * np.random.randn(n)
Rvv = 9e-2
vts = math.sqrt(Rvv) * np.random.randn(n)

def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

vfA = np.vectorize(fx)

def fy(x, v):
    return x ** 2 + x ** 3 + v

def fc(y, xi):
    return -.5 * math.log(2 * math.pi * Rvv) - (y - xi**2 - xi**3)**2 / (2 * Rvv)

vfC = np.vectorize(fc)

xts = np.zeros((n,))
xts[0] = 2.
yts = np.zeros((n,))
yts[0] = fy(xts[0], vts[0])
for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])

wits = math.sqrt(Rww) * np.random.randn(n, nsamp)
xits = np.zeros((n, nsamp))
xits[0, :] = xts[0] + math.sqrt(1e-20) * np.random.randn(nsamp)

Wits = np.zeros((n, nsamp))

for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])

    xits[tk, :] = vfA(xits[tk - 1, :], wits[tk - 1, :])

    Wi = vfC(yts[tk], xits[tk, :])
    Wits[tk, :] = Wi / sum(Wi)



pass