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
        # self.vAcurl = np.vectorize(self.Acurl)
        # self.vCcurl = np.vectorize(self.Ccurl)
        self.nsamp = 250

    def a(self, x, w=0):
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + w

    def c(self, x, v=0):
        return x ** 2 + x ** 3 + v

    def A(self, x):
        return 1 - .05 * self.dt + .08 * self.dt * x

    def C(self, x):
        return 2 * x + 3 * x ** 2

    def X(self, xhat, Ptil):
        return [xhat, xhat + math.sqrt(self.k0 * Ptil), xhat - math.sqrt(self.k0 * Ptil)]

    def Xhat(self, X, Rww):
        return [X[0], X[1] + self.kappa * math.sqrt(Rww), X[2] - self.kappa * math.sqrt(Rww)]

    def Apf(self, x):
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + math.sqrt(self.Rww) * np.random.randn(self.nsamp)

    def Cpf(self, y, x):
        return np.exp(-np.log(2. * np.pi * self.Rvv) / 2. - (y - x**2 - x**3)**2 / (2. * self.Rvv))

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
        self.tsteps = 1501
        self.dt = .01
        self.x = np.array([2., .05, .04])
        self.Rww = 1e-9 * np.array([1, 1, 1])
        self.Rvv = 9e-2
        self.log = []
        self.n = 3
        self.kappa = 1
        self.alpha = 1
        self.beta = 2
        self.nk = self.n + self.kappa
        self.lam = self.alpha**2 * self.nk - self.n
        self.nl = self.n + self.lam
        self.k1 = self.lam / float(self.nl)
        self.k2 = 1 / float(2 * self.nl)
        self.W0y = self.lam / float(self.nl)
        self.W0 = self.lam / float(self.nl) + (1 - self.alpha**2 + self.beta)
        self.nsamp = 250

    def a(self, x, w=0):
        return np.array([(1 - x[1] * self.dt) * x[0] + x[2] * self.dt * x[0] ** 2, x[1], x[2]]) + w

    def c(self, x, v=0):
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
        C = np.linalg.cholesky(P)
        X = np.zeros([3, 7])
        X[:, 0] = x
        X[:, 1] = x + math.sqrt(self.nl) * C.T[:, 0]
        X[:, 2] = x + math.sqrt(self.nl) * C.T[:, 1]
        X[:, 3] = x + math.sqrt(self.nl) * C.T[:, 2]
        X[:, 4] = x - math.sqrt(self.nl) * C.T[:, 0]
        X[:, 5] = x - math.sqrt(self.nl) * C.T[:, 1]
        X[:, 6] = x - math.sqrt(self.nl) * C.T[:, 2]
        return X

    def Xhat(self, X, Rww):
        Xhat = np.zeros([3, 7])
        Xhat[:, 0] = X[:, 0]
        Xhat[:, 1] = X[:, 1] + np.array([self.kappa * math.sqrt(Rww[0]), 0, 0])
        Xhat[:, 2] = X[:, 2] + np.array([0, self.kappa * math.sqrt(Rww[1]), 0])
        Xhat[:, 3] = X[:, 3] + np.array([0, 0, self.kappa * math.sqrt(Rww[2])])
        Xhat[:, 4] = X[:, 4] - np.array([self.kappa * math.sqrt(Rww[0]), 0, 0])
        Xhat[:, 5] = X[:, 5] - np.array([0, self.kappa * math.sqrt(Rww[1]), 0])
        Xhat[:, 6] = X[:, 6] - np.array([0, 0, self.kappa * math.sqrt(Rww[2])])
        return Xhat
        # C = np.linalg.cholesky(Rww)
        # Xhat[:, 1] = X[:, 1] + self.nl * C.T[:, 0]
        # Xhat[:, 2] = X[:, 2] + self.nl * C.T[:, 1]
        # Xhat[:, 3] = X[:, 3] + self.nl * C.T[:, 2]
        # Xhat[:, 4] = X[:, 4] + self.nl * C.T[:, 0]
        # Xhat[:, 5] = X[:, 5] + self.nl * C.T[:, 1]
        # Xhat[:, 6] = X[:, 6] + self.nl * C.T[:, 2]

    def va(self, X):
        for i in range(7): X[:, i] = self.a(X[:, i], 0)
        return X

    def vc(self, Xhat):
        Y = np.zeros(7)
        for i in range(7): Y[i] = self.c(Xhat[:, i], 0)
        return Y

    def Xtil(self, X, W):
        Xtil = np.zeros((3, 7))
        for i in range(7): Xtil[:, i] = X[:, i] - W @ X.T
        return Xtil

    def ksi(self, Y, W):
        ksi = np.zeros((1, 7))
        for i in range(7): ksi[0, i] = Y[i] - W @ Y.T
        return ksi

    def Apf(self, x):
        return (1 - x[1] * self.dt) * x[0] + x[2] * self.dt * x[0] ** 2

    def Cpf(self, y, x):
        return np.exp(-np.log(2. * np.pi * self.Rvv) / 2. - (y - x**2 - x**3)**2 / (2. * self.Rvv))

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

