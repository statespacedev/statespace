import math
import numpy as np
from modelbase import ModelBase, SPKFBase, PFBase

class Threestate(ModelBase):
    '''three-state reference model'''

    def ekf(self): return self.steps, self.f, self.h, self.F, self.H, self.R, self.x0, self.P0
    def ekfud(self): return self.G, self.Q
    def sp(self): return self.SPKF.vf, self.SPKF.vh, self.SPKF.X1, self.SPKF.X2, self.SPKF.W, self.SPKF.S
    def pf(self): return self.PF.nsamp, self.PF.F, self.PF.H

    def __init__(self):
        super().__init__()
        self.tsteps = 1501
        self.dt = .01
        self.x = np.array([2., .05, .04])
        self.Rww = 1e-9 * np.array([1, 1, 1])
        self.R = 9e-2
        self.x0 = np.array([2, .055, .044])
        self.P0 = 1. * np.eye(3)
        self.G = np.eye(3)
        self.Q = np.diag(self.Rww)
        self.SPKF = SPKF(self)
        self.PF = PF(self)

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            self.x = self.f(self.x)
            self.y = self.h(self.x)
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

    def f(self, x):
        w = np.multiply(np.random.randn(1, 3), np.sqrt(np.diag(self.Rww)))
        return np.array([(1 - x[1] * self.dt) * x[0] + x[2] * self.dt * x[0] ** 2, x[1], x[2]]) + np.diag(w)

    def h(self, x):
        v = math.sqrt(self.R) * np.random.randn()
        return x[0] ** 2 + x[0] ** 3 + v

    def F(self, x):
        A = np.eye(3)
        A[0, 0] = 1 - x[1] * self.dt + 2 * x[2] * self.dt * x[0]
        A[0, 1] = -self.dt * x[0]
        A[0, 2] = self.dt * x[0] ** 2
        return A

    def H(self, x):
        return np.array([2 * x[0] + 3 * x[0] ** 2, 0, 0])

class SPKF(SPKFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        P0 = .1 * np.eye(3)
        self.S = np.linalg.cholesky(P0)
        self.Sw = np.linalg.cholesky(np.diag(parent.Rww))
        self.Sv = np.linalg.cholesky(np.diag(parent.R * np.array([1])))
        n = 3
        kappa = 1
        alpha = 1
        beta = 2
        lam = alpha ** 2 * (n + kappa) - n
        wi = 1 / float(2 * (n + lam))
        w0m = lam / float(n + lam)
        w0c = lam / float(n + lam) + (1 - alpha ** 2 + beta)
        self.Wm = np.array([w0m, wi, wi, wi, wi, wi, wi])
        self.Wc = np.array([w0c, wi, wi, wi, wi, wi, wi])
        self.nlroot = math.sqrt(n + lam)
        self.Xtil = np.zeros((3, 7))
        self.Ytil = np.zeros((1, 7))
        self.Pxy = np.zeros((3, 1))
        self.W = self.Wm

    def X1(self, x, C):
        X = np.zeros([3, 7])
        X[:, 0] = x
        X[:, 1] = x + self.nlroot * C.T[:, 0]
        X[:, 2] = x + self.nlroot * C.T[:, 1]
        X[:, 3] = x + self.nlroot * C.T[:, 2]
        X[:, 4] = x - self.nlroot * C.T[:, 0]
        X[:, 5] = x - self.nlroot * C.T[:, 1]
        X[:, 6] = x - self.nlroot * C.T[:, 2]
        return X

    def X2(self, X):
        Xhat = np.zeros([3, 7])
        Xhat[:, 0] = X[:, 0]
        Xhat[:, 1] = X[:, 1] + self.nlroot * self.Sw.T[:, 0]
        Xhat[:, 2] = X[:, 2] + self.nlroot * self.Sw.T[:, 1]
        Xhat[:, 3] = X[:, 3] + self.nlroot * self.Sw.T[:, 2]
        Xhat[:, 4] = X[:, 4] - self.nlroot * self.Sw.T[:, 0]
        Xhat[:, 5] = X[:, 5] - self.nlroot * self.Sw.T[:, 1]
        Xhat[:, 6] = X[:, 6] - self.nlroot * self.Sw.T[:, 2]
        return Xhat

    def vf(self, X):
        for i in range(7): X[:, i] = self.parent.f(X[:, i])
        return X

    def vh(self, Xhat):
        Y = np.zeros(7)
        for i in range(7): Y[i] = self.parent.h(Xhat[:, i])
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

if __name__ == "__main__":
    pass

