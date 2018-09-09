# extended bayesian processor, extended kalman filter, XBP, EKF
import numpy as np
import util, math
import class_innov

n = 150
deltat = .01
tts = np.arange(0, n * deltat, deltat)

Rww = 0
wts = math.sqrt(Rww) * np.random.randn(n)
xts = np.zeros(n)
xts[0] = 2.
def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])

Rvv = .09
vts = math.sqrt(Rvv) * np.random.randn(n)
yts = np.zeros(n)
def fy(x, v):
    return x**2 + x**3 + v
yts[0] = fy(xts[0], vts[0])
for tk in range(1, n):
    yts[tk] = fy(xts[tk], vts[tk])

xhatts = np.zeros(n)
xhatts[0] = 2.2
def fA(x):
    return 1 - .05 * deltat + .08 * deltat * x
Ptilts = np.zeros(n)
#Ptilts[tk] = fA(xhatts[tk - 1]) ** 2 * Ptilts[tk - 1]
Ptilts[0] = .01

tk = 0
yhatts = np.zeros(n)
yhatts[tk] = fy(xhatts[tk], 0)
ets = np.zeros(n)
ets[tk] = yts[tk] - yhatts[tk]
def fC(x):
    return 2 * x + 3 * x**2
C = fC(xhatts[tk])
Reets = np.zeros(n)
Reets[tk] = C * Ptilts[tk] * C + Rvv

Kts = np.zeros(n)
Kts[tk] = Ptilts[tk] * C / Reets[tk]
xtilts = np.zeros(n)
xtilts[0] = xts[0] - xhatts[0]

for tk in range(1, n):
    xhatts[tk] = fx(xhatts[tk-1], 0)
    Ptilts[tk] = fA(xhatts[tk-1])**2 * Ptilts[tk-1]
    yhatts[tk] = fy(xhatts[tk], 0)
    ets[tk] = yts[tk] - yhatts[tk]
    C = fC(xhatts[tk])
    Reets[tk] = C * Ptilts[tk] * C + Rvv
    Kts[tk] = Ptilts[tk] * C / Reets[tk]
    xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
    Ptilts[tk] = (1 - Kts[tk] * C) * Ptilts[tk]
    xtilts[tk] = xts[tk] - xhatts[tk]

innov = class_innov.Innov(tts, ets)
innov.standard(tts, xhatts, xtilts, yhatts)


