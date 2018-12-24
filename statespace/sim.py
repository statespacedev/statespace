import math
import numpy as np

class Sim1():
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = 2.
        self.Rww = 0
        self.Rvv = .09
        self.log = []

    def steps(self):
        for tstep in range(self.tsteps):
            tsec = tstep * self.dt
            w = math.sqrt(self.Rww) * np.random.randn()
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = (1 - .05 * self.dt) * self.x + (.04 * self.dt) * self.x**2 + w
            self.y = self.x**2 + self.x**3 + v
            self.log.append([tsec, self.x, self.y])
            if tsec == 0: continue
            yield (tsec, self.x, self.y)

if __name__ == "__main__":
    sim = Sim1()
    for step in sim.steps():
        print(step)

