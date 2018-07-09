# linear bayesian processor, linear kalman filter

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
Rvv = .09**2
vts = math.sqrt(Rvv) * np.random.randn(n)
Rww = 0
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros_like(tts)
yts = np.zeros_like(tts)
xrefts = np.zeros_like(tts)
xhatts = np.zeros_like(tts)
Ptilts = np.zeros_like(tts)
Reets = np.zeros_like(tts)
Kts = np.zeros_like(tts)
yhatts = np.zeros_like(tts)
ets = np.zeros_like(tts)
xtilts = np.zeros_like(tts)

xts[0] = 2. + wts[0]
yts[0] = y(xts[0], vts[0])
xrefts[0] = 2.
xhatts[0] = 2.3
Ptilts[0] = .01
Reets[0] = 0
Kts[0] = 0
yhatts[0] = (xrefts[0]**2 + xrefts[0]**3) + C(xrefts[0]) * (xhatts[0] - xrefts[0])
ets[0] = 0
xtilts[0] = xhatts[0] - xts[0]

for tk in range(1, n):
    xts[tk] = x(xts[tk - 1], wts[tk - 1])
    yts[tk] = y(xts[tk], vts[tk])

    xrefts[tk] = 2. + .067 * tts[tk]
    xhatts[tk] = (1 - .05 * deltat) * xrefts[tk-1] + A(xrefts[tk-1]) * (xhatts[tk-1] - xrefts[tk-1])
    Ptilts[tk] = A(xrefts[tk-1])**2 * Ptilts[tk-1]

    Reets[tk] = C(xhatts[tk])**2 * Ptilts[tk] + .09
    Kts[tk] = util.div0( Ptilts[tk] * C(xrefts[tk]) , Reets[tk] )

    yhatts[tk] = y(xrefts[tk], 0) + C(xrefts[tk]) * (xhatts[tk] - xrefts[tk])
    ets[tk] =  yts[tk] - yhatts[tk]

    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = (1 - Kts[tk] * C(xrefts[tk])) * Ptilts[tk]

    xtilts[tk] = xhatts[tk] - xts[tk]

plots.xts(ets, tts, end=None)

pass

