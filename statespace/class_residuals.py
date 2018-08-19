import numpy as np
import matplotlib.pyplot as plt
import math

class Residuals():
    def __init__(self, tts, ets):
        self.tts = tts
        self.ets = ets
        self.zmw()
        return

    def autocorr1(self, x):
        res = np.zeros_like(x)[1:]
        n = x.size
        for k in range(1, n): # k = 1 to n-1
            res[k-1] = 0
            for t in range(1, n-k): # t = 1 to n-k
                res[k-1] += (self.ets[t-1] - self.mhate) * (self.ets[t-1+k] - self.mhate) #/ (n-k)
        return res

    def autocorr2(self, x):
        res = np.correlate(x, x, mode='full')
        return res[res.size//2:]

    def zmw(self):
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        n = len(self.ets)
        self.tau = 1.96 * math.sqrt(self.Rhatee / n)
        self.iszeromean = True
        if abs(self.mhate) > self.tau:
            self.iszeromean = False
        test1 = self.autocorr1(self.ets)
        test2 = self.autocorr2(self.ets)
        self.Reets = test2
        return self.Reets

    def wssr(self):
        return

    def standard(self, tts, xhatts, xtilts, yhatts):
        lw = 1
        plt.subplot(3, 2, 1)
        plt.plot(tts, xhatts, linewidth=lw)
        plt.subplot(3, 2, 2)
        plt.plot(tts, xtilts, linewidth=lw)
        plt.subplot(3, 2, 3)
        plt.plot(tts, yhatts, linewidth=lw)
        plt.subplot(3, 2, 4)
        plt.plot(tts, self.ets, linewidth=lw)
        plt.subplot(3, 2, 5)
        plt.plot(tts, self.Reets, linewidth=lw)
        plt.show()