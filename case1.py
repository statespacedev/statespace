
# linear bayesian processor, linear kalman filter

import numpy as np
import util, math, plots
n = 150
deltat = .01

def xref(t):
    return 2. + .067 * t

def x(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 + x**3 + v

def A(x):
    return 1 - .05 * deltat + .08 * deltat * x

def C(x):
    return 2 * x + 3 * x**2

Rvv = .09**2
Rww = 0
tts = np.arange(0, n * deltat, deltat)
vts = math.sqrt(Rvv) * np.random.randn(n)
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros_like(tts)
xts[0] = 2.
yts = np.zeros_like(tts)
yts[0] = y(xts[0], vts[0])

xrefts = np.zeros_like(tts)
xrefts[0] = xref(t=0)

xhatts = np.zeros_like(tts)
xhatts[0] = 2.3
xtilts = np.zeros_like(tts)
xtilts[0] = xhatts[0] - xts[0]
Ptilts = np.zeros_like(tts)
Ptilts[0] = .01

yhatts = np.zeros_like(tts)
yhatts[0] = (xrefts[0]**2 + xrefts[0]**3) + (2 * xrefts[0] + 3 * xrefts[0]**2) * (xhatts[0] - xrefts[0])
ets = np.zeros_like(tts)
Reets = np.zeros_like(tts)
Kts = np.zeros_like(tts)

for tk in range(1, n):
    xrefts[tk] = xref(tts[tk])

    xts[tk] = x(xts[tk - 1], wts[tk - 1])
    yts[tk] = y(xts[tk], vts[tk])

    xhatts[tk] = (1 - .05 * deltat) * xrefts[tk-1] + (1 - .05 * deltat + .08 * deltat * xrefts[tk-1]) * (xhatts[tk-1] - xrefts[tk-1])
    Ptilts[tk] = (1 - .05 * deltat + .08 * deltat * xrefts[tk-1])**2 * Ptilts[tk-1]

    Reets[tk] = (2 * xhatts[tk] + 3 * xhatts[tk]**2)**2 * Ptilts[tk] + .09
    Kts[tk] = util.div0( Ptilts[tk] * (2 * xrefts[tk] + 3 * xrefts[tk]**2) , Reets[tk] )

    yhatts[tk] = (xrefts[tk]**2 + xrefts[tk]**3) + (2 * xrefts[tk] + 3 * xrefts[tk]**2) * (xhatts[tk] - xrefts[tk])
    ets[tk] =  yts[tk] - yhatts[tk]

    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = (1 - Kts[tk] * (2 * xrefts[tk] + 3 * xrefts[tk]**2)) * Ptilts[tk]

    xtilts[tk] = xhatts[tk] - xts[tk]

plots.xts(xhatts, tts)

pass

