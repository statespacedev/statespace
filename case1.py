
# linear bayesian processor, linear kalman filter

import numpy as np
import util, math, plots
n = 150
deltat = .01

def xref(t):
    return 2. + .067 * t

def yref(t):
    return xref(t)**2 + xref(t)**3

def x(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 + x**3 + v

def yhat(yref, xref, xhat):
    return yref + (2 * xref + 3 * xref**2) * (xhat - xref)

def A(x):
    return 1 - .05 * deltat + .08 * deltat * x

def C(x):
    return 2 * x + 3 * x**2

def xst(xref, xhat):
    return (1 - .05 * deltat) * xref + (1 - .05 * deltat + .08 * deltat * xref) * (xhat - xref)

def Pst(xref, Ptil):
    return (1 - .05 * deltat + .08 * deltat * xref)**2 * Ptil

def Ree(xhat, Ptil):
    return (2 * xhat + 3 * xhat**2)**2 * Ptil + .09

def K(Ptil, xref, Ree):
    return util.div0( Ptil * (2 * xref + 3 * xref**2) , Ree )

def xup(xhat, K, e):
    return xhat + K * e

def Pup(K, xref, Ptil):
    return (1 - K * (2 * xref + 3 * xref**2)) * Ptil

Rvv = .09**2
Rww = 0
tts = np.arange(0, n * deltat, deltat)
vts = math.sqrt(Rvv) * np.random.randn(n)
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros_like(tts)
xts[0] = 2.3
yts = np.zeros_like(tts)
yts[0] = y(xts[0], vts[0])

xrefts = np.zeros_like(tts)
xrefts[0] = xref(t=0)
yrefts = np.zeros_like(tts)
yrefts[0] = yref(t=0)

xhatts = np.zeros_like(tts)
xhatts[0] = 2.3
xtilts = np.zeros_like(tts)
xtilts[0] = xhatts[0] - xts[0]
Ptilts = np.zeros_like(tts)
Ptilts[0] = .01

yhatts = np.zeros_like(tts)
yhatts[0] = yhat(yrefts[0], xrefts[0], xhatts[0])
ets = np.zeros_like(tts)
Reets = np.zeros_like(tts)
Kts = np.zeros_like(tts)

for tk in range(1, n):
    xts[tk] = x(xts[tk - 1], wts[tk - 1])
    yts[tk] = y(xts[tk], vts[tk])
    xrefts[tk] = xref(tts[tk])
    yrefts[tk] = yref(tts[tk])

    xhatts[tk] = xst(xrefts[tk - 1], xhatts[tk - 1])
    Ptilts[tk] = Pst(xrefts[tk - 1], Ptilts[tk - 1])
    yhatts[tk] = yhat(yrefts[tk], xrefts[tk], xhatts[tk])

    ets[tk] =  yts[tk] - yhatts[tk]
    Reets[tk] = Ree(xhatts[tk], Ptilts[tk])
    Kts[tk] = K(Ptilts[tk], xrefts[tk], Reets[tk])

    xhatts[tk] = xup(xhatts[tk], Kts[tk], ets[tk])
    Ptilts[tk] = Pup(Kts[tk], xrefts[tk], Ptilts[tk])

    xtilts[tk] = xhatts[tk] - xts[tk]

plots.xts(ets, tts)

pass

