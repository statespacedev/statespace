# sigma-point bayesian processor, unscented kalman filter, SPBP, UKF
import numpy as np
import util, math, plots

n = 150
deltat = .01

def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
vfx = np.vectorize(fx)
def vfa(x):
    return vfx(x, 0)
def fy(x, v):
    return x**2 + x**3 + v
vfy = np.vectorize(fy)
def vfc(x):
    return vfy(x, 0)

tts = np.arange(0, n * deltat, deltat)
Rvv = .09
vts = math.sqrt(Rvv) * np.random.randn(n)
Rww = 0
wts = math.sqrt(Rww) * np.random.randn(n)
xts = np.zeros((n,))
xts[0] = 2.
yts = np.zeros((n,))
yts[0] = fy(xts[0], vts[0])

xhat0 = 2.1
Ptil0 = .01
'''weights'''
bignsubx = 1
kappa = 1
bkfac = kappa / float(bignsubx + kappa)
W = np.zeros((3,))
W[:] = [bkfac, .5 * bkfac, .5 * bkfac]
'''state prediction'''
def X(xhat, Ptil):
    a = math.sqrt((bignsubx + kappa) * Ptil)
    return [xhat, xhat + a, xhat - a]
Xts = np.zeros((n, 3))
Xts[0, :] = X(xhat0, Ptil0) # Xts[tk, :] = a(Xts[tk-1, :]) + b(uts[tk-1, :])
xhatts = np.zeros((n,))
xhatts[0] = xhat0 # xhatts[tk] = W @ Xts[tk, :]
'''state error prediction'''
Xtilts = np.zeros((n, 3))
Xtilts[0, :] = Xts[0, :] - xhatts[0] # Xtilts[tk, :] = Xts[tk, :] - xhatts[tk]
Ptilts = np.zeros((n,))
Ptilts[0] = Ptil0 # Ptilts[tk] = W @ Xtilts[tk, :] * np.ones((3,)) @ Xtilts[tk, :] + Rww
'''measurement sigma-points and weights'''
def Xhat(X, Rww):
    a = kappa * math.sqrt(Rww)
    return [X[0], X[1] + a, X[2] - a]
Xhatts = np.zeros((n, 3))
Xhatts[0, :] = Xts[0, :] # Xhatts[tk, :] = Xhat(Xts[tk, :], Rww)
'''measurement prediction'''
Yts = np.zeros((n, 3))
Yts[0, :] = vfc(Xhatts[0, :]) # Yts[tk, :] = c(Xhatts[tk, :])
yhatts = np.zeros((n,))
yhatts[0] = W @ Yts[0, :] # yhatts[tk] = W @ Yts[tk, :]
'''residual prediction'''
ksits = np.zeros((n, 3))
ksits[0, :] = Yts[0, :] - yhatts[0] # ksits[tk, :] = Yts[tk, :] - yhatts[tk]
Rksiksits = np.zeros((n,))
Rksiksits[0] = W @ ksits[0, :] * np.ones((3,)) @ ksits[0, :] + Rvv # Rksiksits[tk] = W @ ksits[tk, :] * np.ones((3,)) @ ksits[tk, :] + Rvv
'''gain'''
RXtilksits = np.zeros((n,))
RXtilksits[0] = W @ Xtilts[0, :] * np.ones((3,)) @ ksits[0,:] # RXtilksits[tk] = W @ Xtilts[tk, :] * np.ones((3,)) @ ksits[tk,:]
Kts = np.zeros((n,))
Kts[0] = RXtilksits[0] / Rksiksits[0] # Kts[tk] = RXtilksits[tk] / Rksiksits[tk]
'''update'''
ets = np.zeros((n,))
ets[0] = yts[0] - yhatts[0] # ets[tk] = yts[tk] - yhatts[tk]
# xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
# Ptilts[tk] = Ptilts[tk] - Kts[tk] * Rksiksits[tk] * Kts[tk]
xtilts = np.zeros((n,))
xtilts[0] = xhatts[0] - xts[0] # xtilts[tk] = xhatts[tk] - xts[tk]
ytilts = np.zeros((n,))
ytilts[0] = yhatts[0] - yts[0] # ytilts[tk] = yhatts[tk] - yts[tk]

for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])
    Xts[tk, :] = vfa(Xts[tk - 1, :]) # + b(uts[tk-1, :])
    xhatts[tk] = W @ Xts[tk, :]
    Xtilts[tk, :] = Xts[tk, :] - xhatts[tk]
    Ptilts[tk] = W @ Xtilts[tk, :] * np.ones((3,)) @ Xtilts[tk, :] + Rww
    Xhatts[tk, :] = Xhat(Xts[tk, :], Rww)
    Yts[tk, :] = vfc(Xhatts[tk, :])
    yhatts[tk] = W @ Yts[tk, :]
    ksits[tk, :] = Yts[tk, :] - yhatts[tk]
    Rksiksits[tk] = W @ ksits[tk, :] * np.ones((3,)) @ ksits[tk, :] + Rvv
    RXtilksits[tk] = W @ Xtilts[tk, :] * np.ones((3,)) @ ksits[tk, :]
    ets[tk] = yts[tk] - yhatts[tk]
    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = Ptilts[tk] - Kts[tk] * Rksiksits[tk] * Kts[tk]
    xtilts[tk] = xhatts[tk] - xts[tk]
    ytilts[tk] = yhatts[tk] - yts[tk]
plots.test(xhatts, xtilts, yhatts, ets, yts, Rksiksits, tts)

pass

