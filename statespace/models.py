import math
import numpy as np

class Jazwinski1():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = 2.
        self.Rww = 1e-6
        self.Rvv = 9e-2
        self.log = []
        self.va = np.vectorize(self.a)
        self.vc = np.vectorize(self.c)
        self.bignsubx = 1
        self.kappa = 1
        self.k0 = self.bignsubx + self.kappa
        self.k1 = self.kappa / float(self.k0)
        self.k2 = 1 / float(2 * self.k0)
        self.vAcurl = np.vectorize(self.Acurl)
        self.vCcurl = np.vectorize(self.Ccurl)
        self.nsamp = 250

    def a(self, x, w):
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + w

    def c(self, x, v):
        return x ** 2 + x ** 3 + v

    def A(self, x):
        return 1 - .05 * self.dt + .08 * self.dt * x

    def C(self, x):
        return 2 * x + 3 * x ** 2

    def X(self, xhat, Ptil):
        return [xhat, xhat + math.sqrt(self.k0 * Ptil), xhat - math.sqrt(self.k0 * Ptil)]

    def Xhat(self, X, Rww):
        return [X[0], X[1] + self.kappa * math.sqrt(Rww), X[2] - self.kappa * math.sqrt(Rww)]

    def Acurl(self, x, w):
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + w

    def Ccurl(self, y, xi):
        return np.exp(-np.log(2. * np.pi * self.Rvv) / 2. - (y - xi**2 - xi**3)**2 / (2. * self.Rvv))


    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            w = math.sqrt(self.Rww) * np.random.randn()
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = self.a(self.x, w)
            self.y = self.c(self.x, v)
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

class Jazwinski2():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([2., .05, .04])
        self.Rww = np.diag([1e-6, 0, 0])
        self.Rvv = 9e-2
        self.log = []
        # self.va = np.vectorize(self.a)
        # self.vc = np.vectorize(self.c)
        self.bignsubx = 3
        self.kappa = 1
        self.k0 = self.bignsubx + self.kappa
        self.k1 = self.kappa / float(self.k0)
        self.k2 = 1 / float(2 * self.k0)

    def a(self, x, w):
        return np.array([(1 - x[1] * self.dt) * x[0] + x[2] * self.dt * x[0] ** 2, x[1], x[2]]) + w

    def c(self, x, v):
        return x[0] ** 2 + x[0] ** 3 + v

    def A(self, x):
        A = np.eye(3)
        A[0, 0] = 1 - x[1] * self.dt + 2 * x[2] * self.dt * x[0]
        A[0, 1] = -self.dt * x[0]
        A[0, 2] = self.dt * x[0] ** 2
        return A

    def C(self, x):
        return np.array([2 * x[0] + 3 * x[0] ** 2, 0, 0])

    def X(self, x, P):
        X = np.zeros([3, 7])
        X[:, 0] = x
        X[:, 1] = x + np.array([math.sqrt(self.k0 * P[0, 0]), 0, 0])
        X[:, 2] = x + np.array([0, math.sqrt(self.k0 * P[1, 1]), 0])
        X[:, 3] = x + np.array([0, 0, math.sqrt(self.k0 * P[2, 2])])
        X[:, 4] = x - np.array([math.sqrt(self.k0 * P[0, 0]), 0, 0])
        X[:, 5] = x - np.array([0, math.sqrt(self.k0 * P[1, 1]), 0])
        X[:, 6] = x - np.array([0, 0, math.sqrt(self.k0 * P[2, 2])])
        return X

    def va(self, X):
        for i in range(7): X[:, i] = self.a(X[:, i], 0)
        return X

    def Xhat(self, X, Rww):
        Xhat = np.zeros([3, 7])
        Xhat[:, 0] = X[:, 0]
        Xhat[:, 1] = X[:, 1] + np.array([self.kappa * math.sqrt(Rww[0, 0]), 0, 0])
        Xhat[:, 2] = X[:, 2] + np.array([0, self.kappa * math.sqrt(Rww[1, 1]), 0])
        Xhat[:, 3] = X[:, 3] + np.array([0, 0, self.kappa * math.sqrt(Rww[2, 2])])
        Xhat[:, 4] = X[:, 4] - np.array([self.kappa * math.sqrt(Rww[0, 0]), 0, 0])
        Xhat[:, 5] = X[:, 5] - np.array([0, self.kappa * math.sqrt(Rww[1, 1]), 0])
        Xhat[:, 6] = X[:, 6] - np.array([0, 0, self.kappa * math.sqrt(Rww[2, 2])])
        return Xhat

    def vc(self, Xhat):
        Y = np.zeros(7)
        for i in range(7):
            Y[i] = self.c(Xhat[:, i], 0)
        return Y

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            w = np.multiply(np.random.randn(1, 3), np.sqrt(np.diag(self.Rww)))
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = self.a(self.x, w[0])
            self.y = self.c(self.x, v)
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

if __name__ == "__main__":
    sim = Jazwinski1()
    for step in sim.steps():
        print(step)

