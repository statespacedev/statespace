# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import util, math, plots

nsamp = 250
n = 150
deltat = .01
Rww = 1e-6
Rvv = 9e-2
x0 = 2.
P0 = 1e-20

def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
def fy(x, v):
    return x**2 + x**3 + v
vfx = np.vectorize(fx)
vfy = np.vectorize(fy)
def vfa(vx):
    return vfx(vx, 0)
def vfc(vx):
    return vfy(vx, 0)

tts = np.arange(0, n * deltat, deltat)
wts = math.sqrt(Rww) * np.random.randn(n)
vts = math.sqrt(Rvv) * np.random.randn(n)
xts = np.zeros((n,))
yts = np.zeros((n,))
xts[0] = x0
yts[0] = fy(xts[0], vts[0])
for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])

xits = np.zeros((n, nsamp))
xits[0, :] = x0 + math.sqrt(P0) * np.random.randn(nsamp)

pass