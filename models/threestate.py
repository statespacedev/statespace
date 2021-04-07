import math
import numpy as np
import matplotlib.pyplot as plt
from basemodel import BaseModel, SPKFBase, PFBase, EvalBase, Autocorr, Log

class Threestate(BaseModel):
    '''three-state reference model'''

    def ekf(self): return self.sim, self.f, self.h, self.F, self.H, self.R, self.Q, self.G, self.x0, self.P0
    def sp(self): return self.SPKF.vf, self.SPKF.vh, self.SPKF.Xtil, self.SPKF.Ytil, \
                         self.SPKF.X1, self.SPKF.X2, self.SPKF.Pxy, \
                         self.SPKF.W, self.SPKF.Wc, self.SPKF.S, self.SPKF.Sproc, self.SPKF.Sobs
    def spcho(self): return self.SPKF.vf, self.SPKF.vh, self.SPKF.Xtil, self.SPKF.Ytil, \
                         self.SPKF.X1cho, self.SPKF.X2cho, self.SPKF.Pxy, \
                         self.SPKF.W, self.SPKF.Wc, self.SPKF.S, self.SPKF.Sproc, self.SPKF.Sobs
    def pf(self): return self.PF.nsamp, self.PF.F, self.PF.H

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

class SPKF(SPKFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        P0 = .1 * np.eye(3)
        n = 3
        kappa = 1
        alpha = 1
        beta = 2
        lam = alpha ** 2 * (n + kappa) - n
        wi = 1 / float(2 * (n + lam))
        w0m = lam / float(n + lam)
        w0c = lam / float(n + lam) + (1 - alpha ** 2 + beta)
        self.nlroot = math.sqrt(n + lam)
        self.W = np.array([[w0m, wi, wi, wi, wi, wi, wi]])
        self.Wc = np.array([[w0c, wi, wi, wi, wi, wi, wi]])
        self.Xtil = np.zeros((3, 7))
        self.Ytil = np.zeros((1, 7))
        self.Pxy = np.zeros((3, 1))
        self.S = np.linalg.cholesky(P0)
        self.Sproc = np.linalg.cholesky(parent.Q)
        self.Sobs = np.linalg.cholesky(np.diag(parent.R * np.array([1])))

    def X1(self, x, P):
        k = 4
        col1 = x
        col2 = x + np.sqrt(k * np.array([[P[0, 0], 0, 0]]).T)
        col3 = x + np.sqrt(k * np.array([[0, P[1, 1], 0]]).T)
        col4 = x + np.sqrt(k * np.array([[0, 0, P[2, 2]]]).T)
        col5 = x - np.sqrt(k * np.array([[P[0, 0], 0, 0]]).T)
        col6 = x - np.sqrt(k * np.array([[0, P[1, 1], 0]]).T)
        col7 = x - np.sqrt(k * np.array([[0, 0, P[2, 2]]]).T)
        X = np.column_stack((col1, col2, col3, col4, col5, col6, col7))
        return X

    def X2(self, X):
        k = 3
        X2 = np.column_stack((X[:, 0].reshape(-1, 1),
                             X[:, 1].reshape(-1, 1) + np.sqrt(k * self.parent.Q[:, 0].reshape(-1, 1)),
                             X[:, 2].reshape(-1, 1) + np.sqrt(k * self.parent.Q[:, 1].reshape(-1, 1)),
                             X[:, 3].reshape(-1, 1) + np.sqrt(k * self.parent.Q[:, 2].reshape(-1, 1)),
                             X[:, 4].reshape(-1, 1) - np.sqrt(k * self.parent.Q[:, 0].reshape(-1, 1)),
                             X[:, 5].reshape(-1, 1) - np.sqrt(k * self.parent.Q[:, 1].reshape(-1, 1)),
                             X[:, 6].reshape(-1, 1) - np.sqrt(k * self.parent.Q[:, 2].reshape(-1, 1))))
        return X2

    def X1cho(self, x, S):
        X = np.column_stack((x,
                             x + self.nlroot * S[:, 0].reshape(-1, 1),
                             x + self.nlroot * S[:, 1].reshape(-1, 1),
                             x + self.nlroot * S[:, 2].reshape(-1, 1),
                             x - self.nlroot * S[:, 0].reshape(-1, 1),
                             x - self.nlroot * S[:, 1].reshape(-1, 1),
                             x - self.nlroot * S[:, 2].reshape(-1, 1)))
        return X

    def X2cho(self, X):
        Xhat = np.zeros([3, 7])
        Xhat[:, 0] = X[:, 0]
        Xhat[:, 1] = X[:, 1] + self.nlroot * self.Sproc.T[:, 0]
        Xhat[:, 2] = X[:, 2] + self.nlroot * self.Sproc.T[:, 1]
        Xhat[:, 3] = X[:, 3] + self.nlroot * self.Sproc.T[:, 2]
        Xhat[:, 4] = X[:, 4] - self.nlroot * self.Sproc.T[:, 0]
        Xhat[:, 5] = X[:, 5] - self.nlroot * self.Sproc.T[:, 1]
        Xhat[:, 6] = X[:, 6] - self.nlroot * self.Sproc.T[:, 2]
        return Xhat

    def vf(self, X):
        for i in range(7):
            tmp = self.parent.f(X[:, i].reshape(-1, 1))
            X[0, i] = tmp[0, 0]
            X[1, i] = tmp[1, 0]
            X[2, i] = tmp[2, 0]
        return X

    def vh(self, Xhat):
        Y = np.zeros((1, 7))
        for i in range(7):
            tmp = Xhat[:, i].reshape(-1, 1)
            tmp2 = self.parent.h(tmp)
            Y[0, i] = self.parent.h(tmp)
        return Y

    def Xtil(self, X, W):
        Xtil = np.zeros((3, 7))
        for i in range(7): Xtil[:, i] = X[:, i] - W @ X.T
        return Xtil

    def ksi(self, Y, W):
        ksi = np.zeros((1, 7))
        for i in range(7): ksi[0, i] = Y[i] - W @ Y.T
        return ksi

class PF(PFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.xhat0 = np.array([2.0, .055, .044])
        self.nsamp = 250

    def F(self, x):
        return (1 - x[1] * self.parent.dt) * x[0] + x[2] * self.parent.dt * x[0] ** 2

    def H(self, y, x):
        return np.exp(-np.log(2. * np.pi * self.parent.R) / 2. - (y - x ** 2 - x ** 3) ** 2 / (2. * self.parent.R))

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

