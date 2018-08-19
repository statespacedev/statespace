# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import math
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
    xtilts = np.zeros((n,))
    xtilts[0] = xts[0] - xhatts[0]

    wits = math.sqrt(Rww) * np.random.randn(n, nsamp)
    xits = np.zeros((n, nsamp))
    xits[0, :] = xhatts[0] + wits[0, :]

    Wits = np.zeros((n, nsamp))
    Wi = vfC(yts[0], xits[0, :])
    Wits[0, :] = Wi / sum(Wi)

    yhatts = np.zeros((n,))
    yhatts[0] = fy(xhatts[0], 0)
    ets = np.zeros((n,))
    ets[0] = yts[0] - yhatts[0]

    for tk in range(1, n):
        xits[tk, :] = vfA(xits[tk - 1, :], wits[tk - 1, :])
        Wi = vfC(yts[tk], xits[tk, :])
        Wits[tk, :] = Wi / sum(Wi)
        xhatts[tk] = Wits[tk, :] @ xits[tk, :]
        xtilts[tk] =  xts[tk] - xhatts[tk]
        yhatts[tk] = fy(xhatts[tk], 0)
        ets[tk] = yts[tk] - yhatts[tk]
        xits[tk, :] = invcdf(xits[tk, :], Wits[tk, :])

    innov = class_residuals.Residuals(tts, ets)
    innov.standard(tts, xhatts, xtilts, yhatts)

def invcdf(xi, Wi):
    tmp = []
    for ndx in range(xi.size):
        tmp.append([xi[ndx], Wi[ndx]])
    tmp = sorted(tmp, key=lambda x: x[0])
    cdf = [[tmp[0][0], tmp[0][1]]]
    cdfndx = 0
    for i in range(1, len(tmp)):
        if abs(tmp[i][0] - tmp[i-1][0]) > 1e-5:
            cdf.append([tmp[i][0], tmp[i][1] + cdf[cdfndx][1]])
            cdfndx += 1
        else:
            cdf[cdfndx][1] += tmp[i][1]
    cdf = np.asarray(cdf)
    uk = np.sort(np.random.uniform(size=xi.size))
    xhati, Whati, k = [], [], 0
    for row in cdf:
        while k < uk.size and uk[k] <= row[1]:
            xhati.append(row[0])
            Whati.append(1/float(xi.size))
            k += 1
    xhati = np.asarray(xhati)
    assert xhati.size == xi.size
    return xhati

if __name__ == "__main__":
    main()
