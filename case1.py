
# linear bayesian processor, linear kalman filter

import numpy as np
import util

def x(x, deltat, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 * x**3 + v

def A(x, deltat):
    return 1 - .05 * deltat + .08 * deltat * x

def C(x):
    return 2 * x + 3 * x**2

def xhat(xref, deltat, xhat):
    return (1 - .05 * deltat) * xref + (1 - .05 * deltat + .08 * xref) * (xhat - xref)

def Ptil(xref, deltat, Ptil):
    return (1 - .05 * deltat + .08 * deltat * xref)**2 * Ptil

def e(y, xref, xhat):
    return y - (xref**2 - xref**3) - (2 * xref + 3 * xref**2) * (xhat - xref)

def Ree(xhat, Ptil):
    return (2 * xhat + 3 * xhat**2)**2 * Ptil + .09

def K(Ptil, xref, Ree):
    return util.div0( Ptil * (2 * xref + 3 * xref**2) , Ree )

n = 150
deltat = .01
tts = np.arange(0, int(n * deltat), deltat)
vts = .09 * np.random.randn(n)
xrefts = .067 * tts+ 2.
Rww = 0

xhatts = np.zeros_like(tts)
xhat[0] = 2.3
Ptilts = np.zeros_like(tts)
Ptilts[0] = .01

xts = np.zeros_like(tts)
yts = np.zeros_like(tts)
ets = np.zeros_like(tts)
Reets = np.zeros_like(tts)
Kts = np.zeros_like(tts)

pass
