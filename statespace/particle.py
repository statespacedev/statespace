# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import math
from performance import Innovations, PMF
import models


def resample(xi, Wi):
    tmp = []
    for i in range(xi.size):
        tmp.append([xi[i], Wi[i]])
    tmp = sorted(tmp, key=lambda x: x[0])
    cdf = [[tmp[0][0], tmp[0][1]]]
    for i in range(1, len(tmp)):
        cdf.append([tmp[i][0], tmp[i][1] + cdf[i - 1][1]])
    cdf = np.asarray(cdf)
    uk = np.sort(np.random.uniform(size=xi.size))
    xhati, k = [], 0
    for row in cdf:
        while k < uk.size and uk[k] <= row[1]:
            xhati.append(row[0])
            k += 1
    xhati = np.asarray(xhati)
    assert xhati.size == xi.size
    return xhati

def roughen(x):
    return x + .1 * np.random.randn(x.size)

def normalize(W):
    return W / sum(W)

class Particle():
    def __init__(self, mode, innov=False, pmfs=False):
        self.innov = Innovations()
        self.pmf = PMF()
        if mode == 'pf1':
            m = models.Jazwinski1()
            self.pf1(m)
        if mode == 'pf2':
            m = models.Jazwinski2()
            self.pf2(m)
        if innov: self.innov.plot()
        if pmfs: self.pmf.plot()

    def pf1(self, m):
        xhat = 2.05
        x = xhat + math.sqrt(1e-5) * np.random.randn(m.nsamp)
        W = normalize(np.ones(m.nsamp))
        for step in m.steps():
            x = resample(x, W)
            x = m.Apf(x)
            W = normalize(m.Cpf(step[2], x))
            yhat = m.c(xhat, 0)
            xhat = W @ x
            self.innov.update([step[0], xhat, yhat, step[1] - xhat, step[2] - yhat])
            self.pmf.update(m, step, x)

    def pf2(self, m):
        xhat = np.array([2.0, .055, .044])
        x = xhat + np.sqrt(m.Rww) * np.random.randn(m.nsamp, 3)
        W = np.ones((m.nsamp, 3)) / m.nsamp
        for step in m.steps():
            x[:, 0], x[:, 1], x[:, 2] = resample(x[:, 0], W[:, 0]), roughen(x[:, 1]), roughen(x[:, 2])
            x[:, 0] = np.apply_along_axis(m.Apf, 1, x)
            W[:, 0] = normalize(m.Cpf(step[2], x[:, 0]))
            yhat = m.c(xhat, 0)
            xhat = [W[:, 0].T @ x[:, 0], W[:, 1].T @ x[:, 1], W[:, 2].T @ x[:, 2]]
            self.innov.update([step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat])

if __name__ == "__main__":
    Particle('pf1', pmfs=0, innov=1)
    Particle('pf2', innov=1)
