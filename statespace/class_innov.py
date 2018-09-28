import numpy as np
import matplotlib.pyplot as plt
import math

class Innov():
    def __init__(self, tts, ets):
        self.tts = tts
        self.ets = ets
        self.zmw()
        return

    def autocorr1(self, x):
        n = x.size
        ac = np.zeros([n, 1])
        for k in range(1, n): # k = 1 to n-1
            ac[k] = 0
            for t in range(1, n-k): # t = 1 to n-k
                ac[k] += (self.ets[t-1] - self.mhate) * (self.ets[t-1+k] - self.mhate) #/ (n-k)
        return ac

    def autocorr2(self, x):
        ac = np.correlate(x, x, mode='full')
        return ac[ac.size//2:]

    def zmw(self):
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        n = len(self.ets)
        self.tau = 1.96 * math.sqrt(self.Rhatee / n)
        self.iszeromean = True
        if abs(self.mhate) > self.tau:
            self.iszeromean = False
        test1 = self.autocorr1(self.ets)
        #test2 = self.autocorr2(self.ets)
        self.Rhatee = test1
        return self.Rhatee

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
        plt.plot(tts, self.Rhatee, linewidth=lw)
        plt.show()

    def abp(self, tts, xhatts, xtilts, yhatts):
        def tsplot(x, y):
            plt.plot(x, y, marker='o', markersize=1, linewidth=0)
        plt.subplot(3, 2, 1)
        tsplot(tts[1:], xhatts[1:, 0])
        plt.subplot(3, 2, 2)
        tsplot(tts[1:], xhatts[1:, 1])
        plt.subplot(3, 2, 3)
        tsplot(tts[1:], xhatts[1:, 2])
        plt.subplot(3, 2, 4)
        tsplot(tts[1:], self.ets[1:])
        plt.subplot(3, 2, 5)
        tsplot(tts[1:], yhatts[1:])
        plt.subplot(3, 2, 6)
        tsplot(tts[1:], self.Rhatee[1:])
        plt.show()
