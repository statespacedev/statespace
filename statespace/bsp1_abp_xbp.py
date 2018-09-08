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

def ver1():
    xhatts[tk, :] = fx(xhatts[tk-1, :], 0)
    Ptilts[tk, :, :] = fA(xhatts[tk-1, :]) @ Ptilts[tk-1, :, :] @ fA(xhatts[tk-1, :]).T + Rww
    yhatts[tk] = fy(xhatts[tk, :], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk, :])
    Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv
    K = Ptilts[tk, :, :] @ C / Reets[tk]
    xhatts[tk, :] = xhatts[tk, :] + K * ets[tk]
    Ptilts[tk, :, :] = (np.eye(3) - K @ C) @ Ptilts[tk, :, :]
    xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

def fUD(P):
    P, L, U = la.lu(P)
    D = np.diag(np.diag(U))   # D is just the diagonal of U
    U /= np.diag(U)[:, None]
    return U, D

def ver2():
    xhatts[tk, :] = fx(xhatts[tk-1, :], 0)
    Ptilts[tk, :, :] = fA(xhatts[tk-1, :]) @ Ptilts[tk-1, :, :] @ fA(xhatts[tk-1, :]).T + Rww
    U, D = fUD(Ptilts[tk, :, :])
    yhatts[tk] = fy(xhatts[tk, :], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk, :])
    Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv
    K = Ptilts[tk, :, :] @ C / Reets[tk]
    xhatts[tk, :] = xhatts[tk, :] + K * ets[tk]
    Ptilts[tk, :, :] = (np.eye(3) - K @ C) @ Ptilts[tk, :, :]
    xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

for tk in range(1, n):
    ver2()

innov = class_residuals.Residuals(tts, ets)
innov.abp(tts, xhatts, xtilts, yhatts)



