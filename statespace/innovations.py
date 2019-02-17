import numpy as np
import matplotlib.pyplot as plt
import math


class Innovations():
    def __init__(self, log, x1=3):
        self.log = np.asarray(log[x1:])
        self.tts = self.log[:, 0]
        self.ets = self.log[:, 4]
        self.genzmw()

    def autocorr1(self, x):
        n = x.size
        ac = np.zeros([n, 1])
        for k in range(1, n):  # k = 1 to n-1
            ac[k] = 0
            for t in range(1, n - k):  # t = 1 to n-k
                ac[k] += (self.ets[t - 1] - self.mhate) * (self.ets[t - 1 + k] - self.mhate)  # / (n-k)
        return ac

    def autocorr2(self, x):
        ac = np.correlate(x, x, mode='full')
        return ac[ac.size // 2:]

    def genzmw(self):
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        n = len(self.ets)
        self.tau = 1.96 * math.sqrt(self.Rhatee / n)
        self.iszeromean = True
        if abs(self.mhate) > self.tau:
            self.iszeromean = False
        self.zmw = self.autocorr1(self.ets)

    def plot_standard(self):
        lw = 1
        plt.subplot(3, 2, 1), plt.plot(self.tts, self.log[:, 1], linewidth=lw), plt.ylabel('xhat')
        plt.subplot(3, 2, 2), plt.plot(self.tts, self.log[:, 3], linewidth=lw), plt.ylabel('err')
        plt.subplot(3, 2, 3), plt.plot(self.tts, self.log[:, 2], linewidth=lw), plt.ylabel('yhat')
        plt.subplot(3, 2, 4), plt.plot(self.tts, self.log[:, 4], linewidth=lw), plt.ylabel('innov')
        plt.subplot(3, 2, 5), plt.plot(self.tts, self.zmw, linewidth=lw), plt.ylabel('autocorr')
        plt.show()
