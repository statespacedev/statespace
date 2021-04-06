import math
import numpy as np
import matplotlib.pyplot as plt
from basemodel import BaseModel, SPKFBase, PFBase, EvalBase, Autocorr, Log

class Onestate(BaseModel):
    '''one-state reference model'''

    def ekf(self): return self.sim, self.f, self.h, self.F, self.H, self.R, self.Q, self.G, self.x0, self.P0
    def sp(self): return self.SPKF.vf, self.SPKF.vh, self.SPKF.Xtil, self.SPKF.Ytil, \
                         self.SPKF.X1, self.SPKF.X2, self.SPKF.Pxy, \
                         self.SPKF.W, self.SPKF.Wc, self.SPKF.S, self.SPKF.Sproc, self.SPKF.Sobs
    def pf(self): return self.PF.nsamp, self.PF.F, self.PF.H

    def __init__(self):
        super().__init__()
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([2.], ndmin=2).T
        self.x0 = np.array([2.2], ndmin=2).T
        self.P0 = .01 * np.eye(1)
        self.varproc = 1e-6
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
            u = np.array([0.])
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
        return np.array(2 * x[0, 0] + 3 * x[0, 0] ** 2, ndmin=2)

class SPKF(SPKFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.kappa = 1
        self.k0 = 1 + self.kappa
        k1 = self.kappa / float(self.k0)
        k2 = 1 / float(2 * self.k0)
        self.W = np.array([[k1, k2, k2]])
        self.Wc = np.array([[k1, k2, k2]])
        self.Xtil = np.zeros((1, 3))
        self.Ytil = np.zeros((1, 3))
        self.Pxy = np.zeros((1, 1))
        self.S = np.linalg.cholesky(parent.P0)
        self.Sproc = np.linalg.cholesky(parent.Q)
        self.Sobs = np.linalg.cholesky(np.diag(parent.R * np.array([1])))

    def X1(self, x, S):
        X = np.column_stack((x,
                             x + self.k0 * S[:, 0].reshape(-1, 1),
                             x - self.k0 * S[:, 0].reshape(-1, 1),))
        return X
        # tmp = np.array([[x[0, 0], x[0, 0] + math.sqrt(self.k0 * S[0, 0]), x[0, 0] - math.sqrt(self.k0 * S[0, 0])]])
        return tmp

    def X2(self, X):
        Xhat = np.zeros([1, 3])
        Xhat[:, 0] = X[:, 0]
        Xhat[:, 1] = X[:, 1] + self.k0 * self.Sproc.T[:, 0]
        Xhat[:, 2] = X[:, 2] - self.k0 * self.Sproc.T[:, 0]
        return Xhat
    # def X2(self, X):
    #     tmp = np.array([[x[0, 0], x[0, 0] + math.sqrt(self.k0 * S[0, 0]), x[0, 0] - math.sqrt(self.k0 * S[0, 0])]])
        tmp = np.array([[X[:, 0], X[:, 1] + self.kappa * math.sqrt(self.parent.varproc), X[:, 2] - self.kappa * math.sqrt(self.parent.varproc)]])
        # return tmp

    # def vf(self): return np.vectorize(self.parent.f)
    #
    # def vh(self): return np.vectorize(self.parent.h)

    def vf(self, X):
        for i in range(3):
            tmp = self.parent.f(X[:, i].reshape(-1, 1))
            X[0, i] = tmp[0, 0]
        return X

    def vh(self, Xhat):
        Y = np.zeros((1, 3))
        for i in range(3):
            tmp = Xhat[:, i].reshape(-1, 1)
            tmp2 = self.parent.h(tmp)
            Y[0, i] = self.parent.h(tmp)
        return Y

class PF(PFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.xhat0 = 2.05
        self.nsamp = 250

    def F(self, x):
        return (1 - .05 * self.parent.dt) * x + (.04 * self.parent.dt) * x ** 2 + math.sqrt(self.parent.varproc) * np.random.randn(self.nsamp)

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

