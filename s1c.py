# sigma-point bayesian processor, unscented kalman filter, SPBP, UKF

import numpy as np
import util, math, plots
n = 150
deltat = .01

def x(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 + x**3 + v

def A(x):
    return 1 - .05 * deltat + .08 * deltat * x

def C(x):
    return 2 * x + 3 * x**2

tts = np.arange(0, n * deltat, deltat)
Rvv = .09
Rww = 0
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

xts[0] = 2.
yts[0] = y(xts[0], vts[0])
xhatts[0] = 2.
Ptilts[0] = .01
Reets[0] = 0
Kts[0] = 0
yhatts[0] = y(xhatts[0], 0)
ets[0] = 0
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

# plots.xts(xhatts, tts, end=None)
plots.test(xhatts, xtilts, yhatts, ets, yts, Reets, tts)

pass

