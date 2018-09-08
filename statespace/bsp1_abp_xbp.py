# adaptive bayesian processor, joint state/parametric processor
import numpy as np
import scipy.linalg as la
import util, math
import class_residuals

n = 1500
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

xhatts = np.zeros([n, 3])
xhatts[0, :] = [2., .055, .044]
def fA(x):
    A = np.eye(3)
    A[0, 0] = 1 - x[1] * deltat + 2 * x[2] * deltat * x[0]
    A[0, 1] = -deltat * x[0]
    A[0, 2] = deltat * x[0]**2
    return A
Ptilts = np.zeros([n, 3, 3])
Ptilts[0, :, :] = 100. * np.eye(3)
xtilts = np.zeros([n, 3])
xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

tk = 0
yhatts = np.zeros(n)
yhatts[tk] = fy(xhatts[tk, :], 0)
ets = np.zeros(n)
ets[tk] = yts[tk] - yhatts[tk]
def fC(x):
    return np.array([2 * x[0] + 3 * x[0]**2, 0, 0])
C = fC(xhatts[tk, :])
Reets = np.zeros(n)
Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv

def ver1(tk):
    xhatts[tk, :] = fx(xhatts[tk-1, :], 0)
    Ptilts[tk, :, :] = fA(xhatts[tk-1, :]) @ Ptilts[tk-1, :, :] @ fA(xhatts[tk-1, :]).T + Rww
    yhatts[tk] = fy(xhatts[tk, :], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk, :])
    Reets[tk] = C @ Ptilts[tk, :, :] @ C.T + Rvv
    K = Ptilts[tk, :, :] @ C.T / Reets[tk]
    xhatts[tk, :] = xhatts[tk, :] + K * ets[tk]
    Ptilts[tk, :, :] = (np.eye(3) - K @ C) @ Ptilts[tk, :, :]
    xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

def ver2(tk):
    xhatts[tk, :] = fx(xhatts[tk-1, :], 0)
    P = Ptilts[tk-1, :, :]
    #P = fA(xhatts[tk-1, :]) @ P @ fA(xhatts[tk-1, :]).T + Rww
    yhatts[tk] = fy(xhatts[tk, :], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk, :])
    z = yhatts[tk]
    R = Rvv
    H = C
    tmp, L, U = la.lu(P)
    D = np.diag(np.diag(U))
    U /= np.diag(U)[:, None]
    x = xhatts[tk, :]
    v = np.zeros(3)
    w = np.zeros(3)
    delta = z
    for j in range(3):
        delta = delta - H[j] * x[j]
        v[j] = H[j]
        if not j == 0:
            for i in range(j):
                v[j] = v[j] + U[i, j] * H[i]
    sigma = R
    for j in range(3):
        nu = v[j]
        v[j] = v[j] * D[j, j]
        w[j] = nu
        if not j == 0:
            for i in range(j):
                tau = U[i, j] * nu
                U[i, j] = U[i, j] - nu * w[i] / sigma
                w[i] = w[i] + tau
        D[j, j] = D[j, j] * sigma
        sigma = sigma + nu * v[j]
        D[j, j] = D[j, j] * sigma
    epsilon = delta / sigma
    for i in range(3):
        x[i] = x[i] + v[i] * epsilon
    xhatts[tk, :] = x
    P = U @ D @ U.T
    Ptilts[tk, :, :] = P
    pass

for tk in range(1, n):
    ver2(tk)

innov = class_residuals.Residuals(tts, ets)
innov.abp(tts, xhatts, xtilts, yhatts)



