# adaptive bayesian processor, joint state/parametric processor, adaptive sigma-point bayesian processor, unscented kalman filter, SPBP, UKF
import numpy as np
import math
import class_innov

n = 150
deltat = .01
tts = np.arange(0, n * deltat, deltat)

Rww = np.diag([0, 0, 0])
wts = np.multiply(np.random.randn(n, 3), np.sqrt(np.diag(Rww)))
xts = np.zeros([n, 3])
xts[0, :] = [2., .05, .04]
def fx(x, w):
    return np.array([(1 - x[1] * deltat) * x[0] + x[2] * deltat * x[0]**2, x[1], x[2]]) + w
for tk in range(1, n):
    xts[tk, :] = fx(xts[tk - 1, :], wts[tk - 1, :])

Rvv = .09
vts = math.sqrt(Rvv) * np.random.randn(n)
yts = np.zeros(n)
def fy(x, v):
    return x[0]**2 + x[0]**3 + v
yts[0] = fy(xts[0, :], vts[0])
for tk in range(1, n):
    yts[tk] = fy(xts[tk, :], vts[tk])

nx, kappa = 3, 1
a, b = kappa/float(nx+kappa), 1/float(2*(nx+kappa))
W = np.array([a, b, b, b, b, b, b])

tk = 0
xhatts = np.zeros([n, 3])
xhatts[tk, :] = [2., .055, .044]
Ptilts = np.zeros([n, 3, 3])
Ptilts[tk, :, :] = 100. * np.eye(3)
def fX(x, P):
    X = np.zeros([7, 3])
    X[0, :] = x
    X[1, :] = x + np.array([math.sqrt((nx + kappa) * P[0, 0]), 0, 0])
    X[2, :] = x + np.array([0, math.sqrt((nx + kappa) * P[1, 1]), 0])
    X[3, :] = x + np.array([0, 0, math.sqrt((nx + kappa) * P[2, 2])])
    X[4, :] = x - np.array([math.sqrt((nx + kappa) * P[0, 0]), 0, 0])
    X[5, :] = x - np.array([0, math.sqrt((nx + kappa) * P[1, 1]), 0])
    X[6, :] = x - np.array([0, 0, math.sqrt((nx + kappa) * P[2, 2])])
    return X
for tk in range(1, 2):
    X = fX(xhatts[tk-1, :], Ptilts[tk-1, :, :])
    def fa(X):
        return fx(X, 0)
    # X = fa(X)
    # xhatts[tk, :] = W @ X
    # Xtil = X - xhatts[tk, :]
    # Ptilts[tk, :] = W @ np.power(Xtil, 2) + Rww

# def fXhat(X, Rww):
#     return [X[0], X[1] + kappa * math.sqrt(Rww), X[2] - kappa * math.sqrt(Rww)]
# Xhat = X
# Xtil = Xhat - xhatts[tk, :]
# def fc(x):
#     return fy(x, 0)
# Y = fc(Xhat)
# yhatts = np.zeros((n,))
# yhatts[0] = W @ Y
# ets = np.zeros(n)
# ets[0] = yts[0] - yhatts[0]
#
# ksits = np.zeros((n, 3))
# ksits[0, :] = Yts[0, :] - yhatts[0]
# Rksiksits = np.zeros((n,))
# Rksiksits[0] = W @ np.power(ksits[0, :], 2) + Rvv
#
# RXtilksits = np.zeros((n,))
# RXtilksits[0] = W @ np.multiply(Xtilts[0, :], ksits[0, :])
# Kts = np.zeros((n,))
# Kts[0] = RXtilksits[0] / Rksiksits[0]
# xtilts = np.zeros([n, 3])
# xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]
#
# for tk in range(1, n):
#     X = vfX(xhatts[tk-1], Ptilts[tk-1])
#     Xts[tk, :] = vfa(X)
#     xhatts[tk] = W @ Xts[tk, :]
#     Xtilts[tk, :] = Xts[tk, :] - xhatts[tk]
#     Ptilts[tk] = W @ np.power(Xtilts[tk, :], 2) + Rww
#
#     Xhatts[tk, :] = Xhat(Xts[tk, :], Rww)
#
#     Yts[tk, :] = vfc(Xhatts[tk, :])
#     yhatts[tk] = W @ Yts[tk, :]
#     ets[tk] = yts[tk] - yhatts[tk]
#
#     ksits[tk, :] = Yts[tk, :] - yhatts[tk]
#     Rksiksits[tk] = W @ np.power(ksits[tk, :], 2) + Rvv
#
#     RXtilksits[tk] = W @ np.multiply(Xtilts[tk, :], ksits[tk, :])
#     Kts[tk] = RXtilksits[tk] / Rksiksits[tk]
#
#     xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
#     Ptilts[tk] = Ptilts[tk] - Kts[tk] * Rksiksits[tk] * Kts[tk]
#     xtilts[tk] = xts[tk] - xhatts[tk]
#
# innov = class_innov.Innov(tts, ets)
# innov.standard(tts, xhatts, xtilts, yhatts)
