# adaptive bayesian processor, joint state/parametric processor, adaptive extended kalman filter
import numpy as np
import util, math
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

tk = 0
xhatts = np.zeros([n, 3])
Ptilts = np.zeros([n, 3, 3])
yhatts = np.zeros(n)
ets = np.zeros(n)
Reets = np.zeros(n)
xtilts = np.zeros([n, 3])

def fA(x):
    A = np.eye(3)
    A[0, 0] = 1 - x[1] * deltat + 2 * x[2] * deltat * x[0]
    A[0, 1] = -deltat * x[0]
    A[0, 2] = deltat * x[0]**2
    return A
xhatts[tk, :] = [2., .055, .044]
Ptilts[tk, :, :] = 100. * np.eye(3)
U, D = util.UD(Ptilts[tk, :, :])
for tmp in range(1, 2):
    Phi = fA(fx(xhatts[tmp-1, :], 0))
    xhat = Phi @ xhatts[tmp-1, :]
    Ptil = Phi @ Ptilts[tmp-1, :, :] @ Phi.T + Rww
    pass

def fC(x):
    return np.array([2 * x[0] + 3 * x[0]**2, 0, 0])
yhatts[tk] = fy(xhatts[tk, :], 0)
ets[tk] = yts[tk] - yhatts[tk]
C = fC(xhatts[tk, :])
Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv
xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

mode = 1
for tk in range(1, n):
    Phi = fA(fx(xhatts[tk-1, :], 0))
    if mode == 0:
        xhatts[tk, :] = Phi @ xhatts[tk-1, :]
        Ptilts[tk, :, :] = Phi @ Ptilts[tk-1, :, :] @ Phi.T + Rww
        yhatts[tk] = fy(xhatts[tk, :], 0)
        ets[tk] = yts[tk] - yhatts[tk]
        C = fC(xhatts[tk, :])
        Reets[tk] = C @ Ptilts[tk, :, :] @ C.T + Rvv
        K = Ptilts[tk, :, :] @ C.T / Reets[tk]
        xhatts[tk, :] = xhatts[tk, :] + K * ets[tk]
        Ptilts[tk, :, :] = (np.eye(3) - K @ C) @ Ptilts[tk, :, :]
    elif mode == 1:
        x, U, D = util.thornton(xin=xhatts[tk-1, :], Phi=Phi, Uin=U, Din=D, Gin=np.eye(3), Q=Rww)
        yhatts[tk] = fy(x, 0)
        ets[tk] = yts[tk] - yhatts[tk]
        x, U, D = util.bierman(z=ets[tk], R=Rvv, H=fC(x), xin=x, Uin=U, Din=D)
        xhatts[tk, :] = x
    xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

innov = class_innov.Innov(tts, ets)
innov.abp(tts, xhatts, xtilts, yhatts)



