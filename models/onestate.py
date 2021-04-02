import math
import numpy as np
import matplotlib.pyplot as plt
from basemodel import BaseModel, SPKFBase, PFBase, EvalBase, Autocorr, Log

class Onestate(BaseModel):
    '''one-state reference model'''

    def ekf(self): return self.sim, self.f, self.h, self.F, self.H, self.R, self.Q, self.G, self.x0, self.P0
    def sp(self): return self.SPKF.vf, self.SPKF.vh, self.SPKF.X1, self.SPKF.X2, self.SPKF.W
    def pf(self): return self.PF.nsamp, self.PF.F, self.PF.H

    def __init__(self):
        super().__init__()
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([2.])
        self.Rww = 1e-6 * np.array([1])
        self.R = 9e-2
        self.x0 = np.array([2.2])
        self.P0 = .01 * np.eye(1)
        self.G = np.eye(1)
        self.Q = self.Rww * np.eye(1)
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
        w = math.sqrt(self.Rww) * np.random.randn()
        base = (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2
        if 0 in args: return base + w
        return base

    def F(self, x):
        A = np.eye(1)
        A[0, 0] = 1 - .05 * self.dt + .08 * self.dt * x
        return A

    def h(self, x, *args):
        v = math.sqrt(self.R) * np.random.randn()
        base = x ** 2 + x ** 3
        if 0 in args: return base + v
        return base

    def H(self, x):
        return 2 * x + 3 * x ** 2

class SPKF(SPKFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.vf = np.vectorize(parent.f)
        self.vh = np.vectorize(parent.h)
        bignsubx = 1
        kappa = 1
        k0 = bignsubx + kappa
        k1 = kappa / float(k0)
        k2 = 1 / float(2 * k0)
        self.W = np.array([k1, k2, k2])
        self.k0 = k0
        self.kappa = kappa

    def X1(self, xhat, Ptil):
        return [xhat, xhat + math.sqrt(self.k0 * Ptil), xhat - math.sqrt(self.k0 * Ptil)]

    def X2(self, X):
        return [X[0], X[1] + self.kappa * math.sqrt(self.parent.Rww), X[2] - self.kappa * math.sqrt(self.parent.Rww)]

class PF(PFBase):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.xhat0 = 2.05
        self.nsamp = 250

    def F(self, x):
        return (1 - .05 * self.parent.dt) * x + (.04 * self.parent.dt) * x ** 2 + math.sqrt(self.parent.Rww) * np.random.randn(self.nsamp)

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
        plt.subplot(3, 2, 1), plt.plot(logm.t, logm.x[:, 0], linewidth=lw), plt.ylabel('x[0]')
        plt.subplot(3, 2, 2), plt.plot(logm.t, logm.y, linewidth=lw), plt.ylabel('y')
        plt.subplot(3, 2, 3), plt.plot(logp.t, logp.x[:, 0], linewidth=lw), plt.ylabel('xe[0]')
        plt.subplot(3, 2, 4), plt.plot(logp.t, logp.y, linewidth=lw), plt.ylabel('ye')
        plt.subplot(3, 2, 5), plt.plot(logp.t, logm.x[:, 0] - logp.x[:, 0], linewidth=lw), plt.ylabel('xe[0] err')
        plt.subplot(3, 2, 6), plt.plot(logp.t, logm.y - logp.y, linewidth=lw), plt.ylabel('ye err')

if __name__ == "__main__":
    pass

