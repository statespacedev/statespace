import numpy as np
import math

class Innov():
    def __init__(self, tts, ets):
        self.tts = tts
        self.ets = ets
        self.autocorrelation()
        return

    def autocorrelation(self):
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        n = len(self.ets)
        self.tau = 1.96 * math.sqrt(self.Rhatee / n)
        self.iszeromean = True
        if abs(self.mhate) > self.tau:
            self.iszeromean = False
        self.Rhatee = []
        for k in range(1, n): # k = 1 to n-1
            tmp = 0
            for t in range(1, n-k+1): # t = 1 to n-k
                tndx = t-1
                tmp += (self.ets[tndx] - self.mhate) * (self.ets[tndx+k] - self.mhate)
            self.Rhatee.append(tmp / (n-k))
        return

    def wssr(self):
        return