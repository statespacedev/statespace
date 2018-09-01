# adaptive bayesian processor, joint state/parametric processor
import numpy as np
import util, math
import class_residuals

n = 150
deltat = .01
tts = np.arange(0, n * deltat, deltat)

Rww = np.array([0, 0, 0])
wts = np.multiply(np.random.randn(n, 3), Rww)
xts = np.zeros([n, 3])
xts[0, :] = [2., .05, .04]
def fx(x, w):
    return [(1 - x[1] * deltat) * x[0] + x[2] * deltat * x[0]**2, x[1], x[2]] + w

Rvv = .09
vts = math.sqrt(Rvv) * np.random.randn(n)
yts = np.zeros([n, 1])
def fy(x, v):
    return x[0]**2 + x[0]**3 + v
yts[0] = fy(xts[0, :], vts[0])

for tk in range(1, n):
    xts[tk, :] = fx(xts[tk - 1, :], wts[tk - 1, :])
    yts[tk] = fy(xts[tk, :], vts[tk])
    pass

# def A(x):
#     return 1 - .05 * deltat + .08 * deltat * x
#
# def C(x):
#     return 2 * x + 3 * x**2


#
# xhatts = np.zeros_like(tts)
# xhatts[0] = 2.2
# Ptilts = np.zeros_like(tts)
# Ptilts[0] = .01
# xtilts = np.zeros_like(tts)
# xtilts[0] = xts[0] - xhatts[0]
#
# Reets = np.zeros_like(tts)
# Reets[0] = 0
# Kts = np.zeros_like(tts)
# Kts[0] = 0
#
# yhatts = np.zeros_like(tts)
# yhatts[0] = y(xhatts[0], 0)
# ets = np.zeros_like(tts)
# ets[0] = yts[0] - yhatts[0]

# for tk in range(1, n):
#     xts[tk] = x(xts[tk - 1], wts[tk - 1])
#     yts[tk] = y(xts[tk], vts[tk])
#
#     xhatts[tk] = x(xhatts[tk-1], 0)
#     Ptilts[tk] = A(xhatts[tk-1])**2 * Ptilts[tk-1]
#
#     Reets[tk] = C(xhatts[tk])**2 * Ptilts[tk] + .09
#     Kts[tk] = util.div0( Ptilts[tk] * C(xhatts[tk]) , Reets[tk] )
#
#     yhatts[tk] = y(xhatts[tk], 0)
#     ets[tk] = yts[tk] - yhatts[tk]
#
#     xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
#     Ptilts[tk] = (1 - Kts[tk] * C(xhatts[tk])) * Ptilts[tk]
#     xtilts[tk] = xts[tk] - xhatts[tk]
#
# innov = class_residuals.Residuals(tts, ets)
# innov.standard(tts, xhatts, xtilts, yhatts)



