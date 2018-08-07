import numpy as np
import math

class Innov():
    def __init__(self, tts, ets):
        self.tts = tts
        self.ets = ets
        self.zeromean()
        return

    def zeromean(self):
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        self.tau = 1.96 * math.sqrt(self.Rhatee / len(self.ets))
        self.iszeromean = True
        if self.mhate > self.tau:
            self.iszeromean = False
        return

    def whiteness(self):
        return

    def wssr(self):
        return