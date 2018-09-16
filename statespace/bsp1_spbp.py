# sigma-point bayesian processor, unscented kalman filter, SPBP, UKF
import numpy as np
import math
import class_innov

n = 150
deltat = .01

xhat0 = 2.2
Ptil0 = .01

bignsubx = 1
kappa = 1
bk = bignsubx + kappa
W = np.zeros((3,))
W[:] = [kappa / float(bk), .5 / float(bk), .5 / float(bk)]

def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def fy(x, v):
    return x ** 2 + x ** 3 + v

vfx = np.vectorize(fx)
vfy = np.vectorize(fy)

def vfa(vx):
    return vfx(vx, 0)

def vfc(vx):
    return vfy(vx, 0)

def vfX(xhat, Ptil):
    return [xhat, xhat + math.sqrt(bk * Ptil), xhat - math.sqrt(bk * Ptil)]

def Xhat(X, Rww):
    return [X[0], X[1] + kappa * math.sqrt(Rww), X[2] - kappa * math.sqrt(Rww)]

tts = np.arange(0, n * deltat, deltat)
Rvv = .09
vts = math.sqrt(Rvv) * np.random.randn(n)
Rww = 0
wts = math.sqrt(Rww) * np.random.randn(n)

xts = np.zeros((n,))
xts[0] = 2.
yts = np.zeros((n,))
yts[0] = fy(xts[0], vts[0])

Xts = np.zeros((n, 3))
Xts[0, :] = vfX(xhat0, Ptil0)
xhatts = np.zeros((n,))
xhatts[0] = xhat0
xtilts = np.zeros((n,))
xtilts[0] = xts[0] - xhatts[0]

Xtilts = np.zeros((n, 3))
Xtilts[0, :] = Xts[0, :] - xhatts[0]
Ptilts = np.zeros((n,))
Ptilts[0] = Ptil0

Xhatts = np.zeros((n, 3))
Xhatts[0, :] = Xts[0, :]

Yts = np.zeros((n, 3))
Yts[0, :] = vfc(Xhatts[0, :])
yhatts = np.zeros((n,))
yhatts[0] = W @ Yts[0, :]
ets = np.zeros((n,))
ets[0] = yts[0] - yhatts[0]

ksits = np.zeros((n, 3))
ksits[0, :] = Yts[0, :] - yhatts[0]
Rksiksits = np.zeros((n,))
Rksiksits[0] = W @ np.power(ksits[0, :], 2) + Rvv

RXtilksits = np.zeros((n,))
RXtilksits[0] = W @ np.multiply(Xtilts[0, :], ksits[0, :])
Kts = np.zeros((n,))
Kts[0] = RXtilksits[0] / Rksiksits[0]

for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])

    X = vfX(xhatts[tk-1], Ptilts[tk-1])
    Xts[tk, :] = vfa(X)
    xhatts[tk] = W @ Xts[tk, :]
    Xtilts[tk, :] = Xts[tk, :] - xhatts[tk]
    Ptilts[tk] = W @ np.power(Xtilts[tk, :], 2) + Rww

    Xhatts[tk, :] = Xhat(Xts[tk, :], Rww)

    Yts[tk, :] = vfc(Xhatts[tk, :])
    yhatts[tk] = W @ Yts[tk, :]
    ets[tk] = yts[tk] - yhatts[tk]

    ksits[tk, :] = Yts[tk, :] - yhatts[tk]
    Rksiksits[tk] = W @ np.power(ksits[tk, :], 2) + Rvv

    RXtilksits[tk] = W @ np.multiply(Xtilts[tk, :], ksits[tk, :])
    Kts[tk] = RXtilksits[tk] / Rksiksits[tk]

    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = Ptilts[tk] - Kts[tk] * Rksiksits[tk] * Kts[tk]
    xtilts[tk] = xts[tk] - xhatts[tk]

innov = class_innov.Innov(tts, ets)
innov.standard(tts, xhatts, xtilts, yhatts)

