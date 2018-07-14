# sigma-point bayesian processor, unscented kalman filter, SPBP, UKF

import numpy as np
import util, math, plots
n = 150
deltat = .01

def x(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 + x**3 + v

def a(x):
    return x(x, 0)

def c(x):
    return y(x, 0)

Rvv = .09
Rww = 0
tts = np.arange(0, n * deltat, deltat)
vts = math.sqrt(Rvv) * np.random.randn(n)
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros((n,))
yts = np.zeros((n,))
xts[0] = 2.
yts[0] = y(xts[0], vts[0])

xhat0 = 2.1
Ptil0 = .01

bignsubx = 1
kappa = 1
bkfac = kappa / float(bignsubx + kappa)
W = np.zeros((3,))
W[:] = [bkfac, bkfac / 2., bkfac / 2.]

def X(xhat, Ptil):
    a = math.sqrt((bignsubx + kappa) * Ptil)
    return [xhat, xhat + a, xhat - a]
Xts = np.zeros((n, 3))
Xts[0, :] = X(xhat0, Ptil0) # Xts[1, :] = a(Xts[0, :]) + b(u)

xhatts = np.zeros((n,))
xhatts[0] = xhat0 # xhatts[1] = W @ Xts[1, :]

Xtilts = np.zeros((n, 3))
Xtilts[0, :] = Xts[0, :] - xhatts[0]

Ptilts = np.zeros((n,))
Ptilts[0] = Ptil0 # Ptilts[1, :] = W @ Xtilts[1, :] * np.ones((3,)) @ Xtilts[1, :] + Rww

def Xhat(X, Rww):
    a = kappa * math.sqrt(Rww)
    return [X, X + a, X - a]
Xhatts = np.zeros((n, 3))
Xhatts[0, :] = Xts[0, :] # Xhatts[1, :] = Xhat(Xts[1, :], Rww)

Yts = np.zeros((n, 3))
Yts[0, :] = c(Xhatts[0, :])

yhatts = np.zeros((n,))
yhatts[0] = W @ Yts[0, :]

ksits = np.zeros((n, 3))
ksits[0, :] = Yts[0, :] - yhatts[0]

Rksiksi = np.zeros((n,))
Rksiksi[0] = W @ ksits[0, :] * np.ones((3,)) @ ksits[0, :] + Rvv

RXtilksi = np.zeros((n,))
RXtilksi[0] = W @ Xtilts[0, :] * np.ones((3,)) @ ksits[0,:]

Kts = np.zeros((n,))
Kts[0] = RXtilksi[0] / Rksiksi[0]

ets = np.zeros((n,))
ets[0] = yts[0] - yhatts[0]

xtilts = np.zeros((n,))
xtilts[0] = xhatts[0] - xts[0]

for tk in range(1, n):
    xts[tk] = x(xts[tk - 1], wts[tk - 1])
    yts[tk] = y(xts[tk], vts[tk])

    xhatts[tk] = x(xhatts[tk-1], 0)
    Ptilts[tk] = A(xhatts[tk-1])**2 * Ptilts[tk-1]

    Reets[tk] = C(xhatts[tk])**2 * Ptilts[tk] + .09
    Kts[tk] = util.div0( Ptilts[tk] * C(xhatts[tk]) , Reets[tk] )

    yhatts[tk] = y(xhatts[tk], 0)
    ets[tk] = yts[tk] - yhatts[tk]

    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = (1 - Kts[tk] * C(xhatts[tk])) * Ptilts[tk]

    xtilts[tk] = xhatts[tk] - xts[tk]

plots.test(xhatts, xtilts, yhatts, ets, yts, Reets, tts)

pass

