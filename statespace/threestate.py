import math
import numpy as np
import matplotlib.pyplot as plt
from statespace.basemodel import BaseModel, SPKFBase, PFBase, EvalBase, Autocorr, Log
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample


class Threestate(BaseModel):
    '''three-state reference model'''

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
        self.tsteps = 1501
        self.dt = .01
        self.x = np.array([[2., .05, .04]]).T
        self.x0 = np.array([[2.1, .055, .044]]).T
        self.P0 = .1 * np.eye(3)
        self.varproc = 1e-9 * np.array([[1, 1, 1]]).T
        self.varobs = 9e-4
        self.R = self.varobs
        self.Q = self.varproc * np.eye(3)
        self.G = np.eye(3)
        self.SPKF = SPKF(self)
        self.PF = PF(self)
        self.eval = Eval(self)

    def sim(self):
        for tstep in range(self.tsteps):
            t = tstep * self.dt
            u = np.array([[0, 0, 0]]).T
            self.x = self.f(self.x, 0) + u
            self.y = self.h(self.x, 0)
            if tstep == 0: continue
            self.log.append([t, self.x, self.y])
            yield (t, self.y, u)

    def f(self, x, *args):
        w = np.multiply(np.random.randn(1, 3), np.sqrt(np.diag(self.Q))).T
        base = np.array([[(1 - x[1, 0] * self.dt) * x[0, 0] + x[2, 0] * self.dt * x[0, 0] ** 2, x[1, 0], x[2, 0]]]).T
        if 0 in args: return base + w
        return base

    def F(self, x):
        A = np.eye(3)
        A[0, 0] = 1 - x[1, 0] * self.dt + 2 * x[2, 0] * self.dt * x[0, 0]
        A[0, 1] = -self.dt * x[0, 0]
        A[0, 2] = self.dt * x[0, 0] ** 2
        return A

    def h(self, x, *args):
        v = math.sqrt(self.R) * np.random.randn()
        base = x[0, 0] ** 2 + x[0, 0] ** 3
        if 0 in args: return base + v
        return base

    def H(self, x):
        return np.array([[2 * x[0, 0] + 3 * x[0, 0] ** 2, 0, 0]])


n, k = 3, 1
w1, w2 = k / (n + k), .5 / (n + k)


class SPKF(SPKFBase):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.W = np.array([[w1, w2, w2, w2, w2, w2, w2]])
        self.WM = np.tile(self.W, (self.parent.x.shape[0], 1))
        P0 = .1 * np.eye(3)
        n = 3
        kappa = 1
        alpha = 1
        lam = alpha ** 2 * (n + kappa) - n
        self.nlroot = math.sqrt(n + lam)
        self.Xtil = np.zeros((3, 7))
        self.Ytil = np.zeros((1, 7))
        self.Pxy = np.zeros((3, 1))
        self.S = np.linalg.cholesky(P0)
        self.Sproc = np.linalg.cholesky(parent.Q)
        self.Sobs = np.linalg.cholesky(np.diag(parent.R * np.array([1])))

    def XY(self, x, P, u):
        k = 4
        col1 = x
        col2 = x + np.sqrt(k * np.array([[P[0, 0], 0, 0]]).T)
        col3 = x + np.sqrt(k * np.array([[0, P[1, 1], 0]]).T)
        col4 = x + np.sqrt(k * np.array([[0, 0, P[2, 2]]]).T)
        col5 = x - np.sqrt(k * np.array([[P[0, 0], 0, 0]]).T)
        col6 = x - np.sqrt(k * np.array([[0, P[1, 1], 0]]).T)
        col7 = x - np.sqrt(k * np.array([[0, 0, P[2, 2]]]).T)
        X = np.column_stack((col1, col2, col3, col4, col5, col6, col7))
        for i in range(7): X[:, i] = (self.parent.f(X[:, i].reshape(-1, 1)) + u).flatten()
        Y = np.zeros((1, 7))
        for i in range(7): Y[0, i] = self.parent.h(X[:, i].reshape(-1, 1)).flatten()
        return X, Y

    def XYcho(self, x, S, u):
        col1 = x
        col2 = x + self.nlroot * S[:, 0].reshape(-1, 1)
        col3 = x + self.nlroot * S[:, 1].reshape(-1, 1)
        col4 = x + self.nlroot * S[:, 2].reshape(-1, 1)
        col5 = x - self.nlroot * S[:, 0].reshape(-1, 1)
        col6 = x - self.nlroot * S[:, 1].reshape(-1, 1)
        col7 = x - self.nlroot * S[:, 2].reshape(-1, 1)
        X = np.column_stack((col1, col2, col3, col4, col5, col6, col7))
        for i in range(7): X[:, i] = (self.parent.f(X[:, i].reshape(-1, 1)) + u).flatten()
        Y = np.zeros((1, 7))
        for i in range(7): Y[0, i] = self.parent.h(X[:, i].reshape(-1, 1)).flatten()
        return X, Y


class PF(PFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.n = 250
        self.x0 = np.array([[2.05, .055, .044]]).T
        self.Q = self.parent.Q
        tmp = 1e-8
        self.Q[1, 1] = tmp
        self.Q[2, 2] = tmp

    def X0(self):
        return self.x0 + np.multiply(np.random.randn(self.x0.shape[0], self.n),
                                     np.diag(np.sqrt(self.parent.Q)).reshape(-1, 1))

    def predict(self, X, u):
        def f(x):
            w = np.multiply(np.random.randn(1, 3), np.sqrt(np.diag(self.Q))).T
            base = np.array([[(1 - x[1] * self.parent.dt) * x[0] + x[2] * self.parent.dt * x[0] ** 2, x[1], x[2]]]).T
            return base + w

        for i in range(X.shape[1]): X[:, i] = f(X[:, i]).flatten()
        return X

    def update(self, X, o):
        W = norm.pdf(X[0, :] ** 2 + X[0, :] ** 3, o, np.sqrt(self.parent.R)).reshape(1, -1)
        return W / np.sum(W)

    def resample(self, x, W):
        ndxs = systematic_resample(W.T)
        return x[:, ndxs]


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
