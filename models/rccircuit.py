import math
import numpy as np
from matplotlib import pyplot as plt


class Rccircuit():
    '''one of the simplest possible reference models from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods.'''
    def __init__(self, signal):
        self.tsteps = 301
        self.dt = .1
        self.x = 2.5 + math.sqrt(1e-6) * np.random.randn()
        self.Rww = 1e-5
        self.Rvv = 4
        self.log = []
        self.custom(signal)

    def custom(self, signal):
        self.u = signal * 1e-6 # amps step-function input
        v = math.sqrt(self.Rvv) * np.random.randn()
        self.y = 2 * self.x + v
        self.log.append([0, self.x, self.y])

    def steps(self):
        for tstep in range(1, self.tsteps):
            tsec = tstep * self.dt
            w = math.sqrt(self.Rww) * np.random.randn()
            v = math.sqrt(self.Rvv) * np.random.randn()
            self.x = .97 * self.x + 100 * self.u + w
            self.y = 2 * self.x + v
            self.log.append([tsec, self.x, self.y])
            yield (tsec, self.x, self.y)
