# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import math
import class_resample
import class_residuals

nsamp = 250
n = 150
deltat = .01

tts = np.arange(0, n * deltat, deltat)
Rww = 1e-6
wts = math.sqrt(Rww) * np.random.randn(n)
Rvv = 9e-2
vts = math.sqrt(Rvv) * np.random.randn(n)

def fx(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
def fy(x, v):
    return x ** 2 + x ** 3 + v

def fA(xi, wi):
    return (1 - .05 * deltat) * xi + .04 * deltat * xi**2 + wi
vfA = np.vectorize(fA)
def fC(y, xi):
    return np.exp(-np.log(2 * np.pi * Rvv) / 2. - (y - xi**2 - xi**3)**2 / (2. * Rvv))
vfC = np.vectorize(fC)

xts = np.zeros((n,))
xts[0] = 2.
yts = np.zeros((n,))
yts[0] = fy(xts[0], vts[0])
for tk in range(1, n):
    xts[tk] = fx(xts[tk - 1], wts[tk - 1])
    yts[tk] = fy(xts[tk], vts[tk])

def main():
    xhatts = np.zeros((n,))
    xhatts[0] = 2.2
    P0 = 1e-20

    xits = np.zeros((n, nsamp))
    xits[0, :] = xhatts[0] + math.sqrt(P0) * np.random.randn(nsamp)
    Wits = np.zeros((n, nsamp))
    Wi = vfC(yts[0], xits[0, :])
    Wits[0, :] = Wi / sum(Wi)

    resamp = class_resample.Resample()
    xhatits = np.copy(xits)
    Whatits = np.copy(Wits)
    xtilts = np.zeros((n,))
    xtilts[0] = xhatts[0] - xts[0]

    yhatts = np.zeros((n,))
    yhatts[0] = fy(xhatts[0], 0)
    ets = np.zeros((n,))
    ets[0] = yts[0] - yhatts[0]

    for tk in range(1, n):
        xits[tk, :] = vfA(xits[tk - 1, :], math.sqrt(Rww) * np.random.randn(nsamp))
        Wi = vfC(yts[tk], xits[tk, :])
        Wits[tk, :] = Wi / sum(Wi)

        xhatits[tk, :], Whatits[tk, :] = resamp.invcdf(xits[tk, :], Wits[tk, :])
        xits[tk, :], Wits[tk, :] = xhatits[tk, :], Whatits[tk, :]

        xhatts[tk] = np.mean(xits[tk, :])
        xtilts[tk] = xhatts[tk] - xts[tk]

        yhatts[tk] = fy(xhatts[tk], 0)
        ets[tk] = yts[tk] - yhatts[tk]

    innov = class_residuals.Residuals(tts, ets)
    innov.standard(tts, xhatts, xtilts, yhatts)

if __name__ == "__main__":
    main()
