import math
import numpy as np

class Sim1():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = 2.
        self.Rww = .000001
        self.Rvv = .09
        self.log = []

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            w = math.sqrt(self.Rww) * np.random.randn()
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = (1 - .05 * self.dt) * self.x + (.04 * self.dt) * self.x**2 + w
            self.y = self.x**2 + self.x**3 + v
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

class Sim1b():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = np.array([2., .05, .04])
        self.Rww = np.diag([0, 0, 0])
        self.Rvv = .09
        self.log = []

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            w = np.multiply(np.random.randn(1, 3), np.sqrt(np.diag(self.Rww)))
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = np.array([(1 - self.x[1]*self.dt)*self.x[0] + self.x[2]*self.dt*self.x[0]**2, self.x[1], self.x[2]]) + w[0]
            self.y = self.x[0]**2 + self.x[0]**3 + v
            if tstep == 0: continue
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)

if __name__ == "__main__":
    sim = Sim1()
    for step in sim.steps():
        print(step)

