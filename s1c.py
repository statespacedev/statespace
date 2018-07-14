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
bignsubx = 1
kappa = 1
tts = np.arange(0, n * deltat, deltat)
vts = math.sqrt(Rvv) * np.random.randn(n)
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros_like(tts)
yts = np.zeros_like(tts)
xhatts = np.zeros_like(tts)
Ptilts = np.zeros_like(tts)
Reets = np.zeros_like(tts)
Kts = np.zeros_like(tts)
yhatts = np.zeros_like(tts)
ets = np.zeros_like(tts)
xtilts = np.zeros_like(tts)
W = np.zeros((3,))
Xts = np.zeros((n, 3))
Xtilts = np.zeros((n, 3))
Xhatts = np.zeros((n, 3))
Yts = np.zeros((n, 3))
ksits = np.zeros((n, 3))

xts[0] = 2.
yts[0] = y(xts[0], vts[0])
xhatts[0] = 2.
Ptilts[0] = .01
Reets[0] = 0
Kts[0] = 0
ets[0] = 0
xtilts[0] = xhatts[0] - xts[0]
W[:] = [kappa / float(bignsubx + kappa), .5 * kappa / float(bignsubx + kappa), .5 * kappa / float(bignsubx + kappa)]
Xts[0, :] = [xhatts[0], xhatts[0] + math.sqrt((bignsubx + kappa) * Ptilts[0]), xhatts[0] - math.sqrt((bignsubx + kappa) * Ptilts[0])]
Xtilts[0, :] = [Xts[0, 0] - xhatts[0], Xts[0, 1] - xhatts[0], Xts[0, 2] - xhatts[0]]
Xhatts[0, :] = Xts[0, :]
Yts[0, :] = c(Xhatts[0, :])
yhatts[0] = W @ Yts[0, :]

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

