# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import math
from innovations import Innovations
import models
from pymc3.plots.kdeplot import fast_kde

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
        self.log = []
        self.pmfs1 = []
        self.pmfs2 = []
        if mode == 'pf1':
            m = models.Jazwinski1()
            self.pf1(m)
        if mode == 'pf2':
            m = models.Jazwinski2()
            self.pf2(m)
        self.innov = Innovations(self.log)
        if innov: self.innov.plot_standard()
        if pmfs: self.pmfs()

    def pf1(self, m):
        xhat = 2.05
        x = xhat + math.sqrt(m.Rww) * np.random.randn(m.nsamp)
        W = normalize(np.ones(m.nsamp))
        for step in m.steps():
            x = resample(x, W)
            x = m.Apf(x)
            W = normalize(m.Cpf(step[2], x))
            yhat = m.c(xhat, 0)
            xhat = W @ x
            self.log.append([step[0], xhat, yhat, step[1] - xhat, step[2] - yhat])
            self.pmfupdate(m, step, x)

    def pmfupdate(self, m, step, x):
        pmf, xmin, xmax = fast_kde(m.Apf(step[1]))
        self.pmfs1.append([step[0], np.linspace(xmin, xmax, len(pmf)), pmf / np.sum(pmf)])
        pmf, xmin, xmax = fast_kde(x)
        self.pmfs2.append([step[0], np.linspace(xmin, xmax, len(pmf)), pmf / np.sum(pmf)])

    def pmfs(self):
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        ax = plt.axes(projection='3d')
        for rec in self.pmfs1:
            y = rec[0] * np.ones(len(rec[1]))
            ax.plot3D(rec[1], y, rec[2], c='g', linewidth=1)
        for rec in self.pmfs2:
            y = rec[0] * np.ones(len(rec[1]))
            ax.plot3D(rec[1], y, rec[2], c='b', linewidth=1)
        plt.show()
        return

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
            self.log.append([step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat])


if __name__ == "__main__":
    Particle('pf1', pmfs=True)
    # Particle('pf2')
