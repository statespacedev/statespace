import math
import numpy as np

class Jazwinski1():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = 2.
        self.Rww = .000001
        self.Rvv = .09
        self.log = []
        self.va = np.vectorize(self.a)
        self.vc = np.vectorize(self.c)
        self.bignsubx = 1
        self.kappa = 1
        self.bk = self.bignsubx + self.kappa

    def a(self, x, w):
        return (1 - .05 * self.dt) * x + (.04 * self.dt) * x ** 2 + w

    def c(self, x, v):
        return x ** 2 + x ** 3 + v

    def A(self, x):
        return 1 - .05 * self.dt + .08 * self.dt * x

    def C(self, x):
        return 2 * x + 3 * x ** 2

    def vfX(self, xhat, Ptil):
        return [xhat, xhat + math.sqrt(self.bk * Ptil), xhat - math.sqrt(self.bk * Ptil)]

    def vfXhat(self, X, Rww):
        return [X[0], X[1] + self.kappa * math.sqrt(Rww), X[2] - self.kappa * math.sqrt(Rww)]

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
        self.Rww = np.diag([0, 0, 0])
        self.Rvv = .09
        self.log = []

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

