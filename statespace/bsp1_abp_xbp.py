# adaptive bayesian processor, joint state/parametric processor
import numpy as np
import util, math
import class_residuals

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
yts = np.zeros([n, 1])
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
#Ptilts[tk, :, :] = fA(xhatts[tk-1, :]) @ Ptilts[tk-1, :, :] @ fA(xhatts[tk-1, :]).T + Rww
Ptilts[0, :, :] = 100. * np.eye(3)

tk = 0
yhatts = np.zeros([n, 1])
yhatts[tk] = fy(xhatts[tk, :], 0)
ets = np.zeros([n, 1])
ets[tk] = yts[tk] - yhatts[tk]
def fC(x):
    return np.array([2 * x[0] + 3 * x[0]**2, 0, 0])
C = fC(xhatts[tk, :])
Reets = np.zeros([n, 1])
Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv

Kts = np.zeros([n, 3])
Kts[tk, :] = Ptilts[tk, :, :] @ C / Reets[tk]
xhatts[tk, :] = xhatts[tk, :] + Kts[tk, :] * ets[tk]
Ptilts[tk, :, :] = (np.eye(3) - Kts[tk, :] @ C) @ Ptilts[tk, :, :]
xtilts = np.zeros([n, 3])
xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

for tk in range(1, n):
    xhatts[tk, :] = fx(xhatts[tk-1, :], 0)
    Ptilts[tk, :, :] = fA(xhatts[tk-1, :]) @ Ptilts[tk-1, :, :] @ fA(xhatts[tk-1, :]).T + Rww
    yhatts[tk] = fy(xhatts[tk, :], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk, :])
    Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv
    Kts[tk, :] = Ptilts[tk, :, :] @ C / Reets[tk]
    xhatts[tk, :] = xhatts[tk, :] + Kts[tk, :] * ets[tk]
    Ptilts[tk, :, :] = (np.eye(3) - Kts[tk, :] @ C) @ Ptilts[tk, :, :]
    xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

innov = class_residuals.Residuals(tts, ets)
innov.standard(tts, xhatts[:, 0], xtilts[:, 0], yhatts)



