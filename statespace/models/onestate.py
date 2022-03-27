import math
import numpy as np
import matplotlib.pyplot as plt
from statespace.models.basemodel import BaseModel, SPKFBase, PFBase, EvalBase, Autocorr, Log
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample


class Onestate(BaseModel):
    '''one-state reference model'''

    def ekf(self):
        return self.sim, self.f, self.h, self.F, self.H, self.R, self.Q, self.G, self.x0, self.P0

    def sp(self):
        return self.SPKF.XY, self.SPKF.W, self.SPKF.WM

    def spcho(self):
        return self.SPKF.XYcho, self.SPKF.W, self.SPKF.Xtil, self.SPKF.Ytil, self.SPKF.Pxy, self.SPKF.S, self.SPKF.Sproc, self.SPKF.Sobs

    def pf(self):
        return self.PF.X0(), self.PF.predict, self.PF.update, self.PF.resample

    def __init__(self):
        super().__init__()
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([[2.]]).T
        self.x0 = np.array([[2.1]]).T
        self.P0 = .1 * np.eye(1)
        self.varproc = 1e-6 * np.array([[1]]).T
        self.varobs = 6e-4
        self.R = self.varobs
        self.Q = self.varproc * np.eye(1)
        self.G = np.eye(1)
        self.SPKF = SPKF(self)
        self.PF = PF(self)
        self.eval = Eval(self)

    def sim(self):
        for tstep in range(self.tsteps):
            t = tstep * self.dt
            u = np.array([0]).T
            self.x = self.f(self.x, 0) + u
            self.y = self.h(self.x, 0)
            if tstep == 0: continue
            self.log.append([t, self.x, self.y])
            yield (t, self.y, u)

    def f(self, x, *args):
        w = math.sqrt(self.varproc) * np.random.randn()
        base = (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2
        if 0 in args: return base + w
        return base

    def F(self, x):
        A = np.eye(1)
        A[0, 0] = 1 - .05 * self.dt + .08 * self.dt * x
        return A

    def h(self, x, *args):
        v = math.sqrt(self.R) * np.random.randn()
        base = x[0, 0] ** 2 + x[0, 0] ** 3
        if 0 in args: return base + v
        return base

    def H(self, x):
        return np.array([[2 * x[0, 0] + 3 * x[0, 0] ** 2]])


n, k = 1, 1
w1, w2 = k / (n + k), .5 / (n + k)


class SPKF(SPKFBase):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.W = np.array([[w1, w2, w2]])
        self.WM = np.tile(self.W, (self.parent.x.shape[0], 1))
        self.kappa = 1
        self.k0 = 1 + self.kappa
        self.nlroot = math.sqrt(self.k0)
        self.S = np.linalg.cholesky(parent.P0)
        self.Sproc = np.linalg.cholesky(parent.Q)
        self.Sobs = np.linalg.cholesky(np.diag(parent.R * np.array([1])))
        self.Xtil = np.zeros((1, 3))
        self.Ytil = np.zeros((1, 3))
        self.Pxy = np.zeros((1, 1))

    def XY(self, x, P, u):
        k = 1
        col1 = x
        col2 = x + np.sqrt(k * np.array([[P[0, 0]]]).T)
        col3 = x - np.sqrt(k * np.array([[P[0, 0]]]).T)
        X = np.column_stack((col1, col2, col3))
        for i in range(3): X[:, i] = (self.parent.f(X[:, i].reshape(-1, 1)) + u).flatten()
        Y = np.zeros((1, 3))
        for i in range(3): Y[0, i] = self.parent.h(X[:, i].reshape(-1, 1)).flatten()
        return X, Y

    def XYcho(self, x, S, u):
        col1 = x
        col2 = x + self.nlroot * S[:, 0].reshape(-1, 1)
        col3 = x - self.nlroot * S[:, 0].reshape(-1, 1)
        X = np.column_stack((col1, col2, col3))
        for i in range(3): X[:, i] = (self.parent.f(X[:, i].reshape(-1, 1)) + u).flatten()
        Y = np.zeros((1, 3))
        for i in range(3): Y[0, i] = self.parent.h(X[:, i].reshape(-1, 1)).flatten()
        return X, Y


class PF(PFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.n = 250
        self.x0 = np.array([[2.05]]).T

    def X0(self):
        return self.x0 + np.multiply(np.random.randn(self.x0.shape[0], self.n),
                                     np.diag(np.sqrt(self.parent.Q)).reshape(-1, 1))

    def predict(self, X, u):
        return (1 - .05 * self.parent.dt) * X + (.04 * self.parent.dt) * X ** 2 + math.sqrt(
            self.parent.varproc) * np.random.randn(self.n)

    def update(self, X, o):
        W = norm.pdf(X ** 2 + X ** 3, o, np.sqrt(self.parent.R))
        return W / np.sum(W)

    def resample(self, x, W):
        return x[:, systematic_resample(W.T)]


class Eval(EvalBase):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.autocorr = Autocorr(parent)

    def estimate(self, proclog):
        lw, logm, logp = 1, Log(self.parent.log), Log(proclog)
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(logm.t, logm.x[:, 0], 'b', linewidth=lw), plt.ylabel('x[0]')
        plt.plot(logp.t, logp.x[:, 0], 'r--', linewidth=lw)
        plt.subplot(2, 2, 2)
        plt.plot(logm.t, logm.y, 'b', linewidth=lw), plt.ylabel('y')
        plt.plot(logm.t, logp.y, 'r--', linewidth=lw)
        plt.subplot(2, 2, 3), plt.plot(logp.t, logm.y - logp.y, 'b', linewidth=lw), plt.ylabel('y err')


if __name__ == "__main__":
    pass
