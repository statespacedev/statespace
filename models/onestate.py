import math
import numpy as np
from modelbase import ModelBase

class Onestate(ModelBase):
    '''one-state reference model.'''

    def pieces(self): return self.steps, self.a, self.c, self.A, self.C, self.Rvv, self.xhat0, self.Ptil0

    def init(self):
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([2.])
        self.Rww = 1e-6 * np.array([1])
        self.Rvv = 9e-2
        self.xhat0 = np.array([2.2])
        self.Ptil0 = .01 * np.eye(1)
        self.G = np.eye(1) # ekfud
        self.Q = self.Rww * np.eye(1) # ekfud
        self.spkf = Spkf(self)
        self.pf = Pf(self)

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            self.x = self.a(self.x)
            self.y = self.c(self.x)
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

    def a(self, x):
        w = math.sqrt(self.Rww) * np.random.randn()
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + w

    def c(self, x):
        v = math.sqrt(self.Rvv) * np.random.randn()
        return x ** 2 + x ** 3 + v

    def A(self, x):
        A = np.eye(1)
        A[0, 0] = 1 - .05 * self.dt + .08 * self.dt * x
        return A

    def C(self, x):
        return 2 * x + 3 * x ** 2

class Spkf():
    def __init__(self, parent):
        self.parent = parent
        self.va = np.vectorize(parent.a)
        self.vc = np.vectorize(parent.c)
        bignsubx = 1
        kappa = 1
        k0 = bignsubx + kappa
        k1 = kappa / float(k0)
        k2 = 1 / float(2 * k0)
        self.W = np.array([k1, k2, k2])
        self.k0 = k0
        self.kappa = kappa

    def X(self, xhat, Ptil):
        return [xhat, xhat + math.sqrt(self.k0 * Ptil), xhat - math.sqrt(self.k0 * Ptil)]

    def Xhat(self, X):
        return [X[0], X[1] + self.kappa * math.sqrt(self.parent.Rww), X[2] - self.kappa * math.sqrt(self.parent.Rww)]

class Pf():
    def __init__(self, parent):
        self.parent = parent
        self.xhat0 = 2.05
        self.nsamp = 250

    def A(self, x):
        return (1 - .05 * self.parent.dt) * x + (.04 * self.parent.dt) * x ** 2 + math.sqrt(self.parent.Rww) * np.random.randn(self.nsamp)

    def C(self, y, x):
        return np.exp(-np.log(2. * np.pi * self.parent.Rvv) / 2. - (y - x ** 2 - x ** 3) ** 2 / (2. * self.parent.Rvv))

if __name__ == "__main__":
    pass

