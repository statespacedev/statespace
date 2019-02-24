import numpy as np
import matplotlib.pyplot as plt
import math

class Ensemble():
    def __init__(self):
        self.log = []

    def update(self, dists):
        for rec in dists.log:
            xmaps = rec[3][np.argmax(rec[4])]
            self.log.append(xmaps)

    def plot(self):
        from pymc3.plots.kdeplot import fast_kde
        import matplotlib.pyplot as plt
        xmappmf, xmapmin, xmapmax = fast_kde(self.log)
        plt.figure()
        plt.plot(np.linspace(xmapmin, xmapmax, len(xmappmf)), xmappmf / np.sum(xmappmf))
        plt.show()
        pass

class Dists():
    def __init__(self):
        self.log = []

    def update(self, m, step, x):
        from pymc3.plots.kdeplot import fast_kde
        pmft, xmint, xmaxt = fast_kde(m.Apf(step[1]))
        pmf, xmin, xmax = fast_kde(x)
        self.log.append([step[0], np.linspace(xmint, xmaxt, len(pmft)), pmft / np.sum(pmft), np.linspace(xmin, xmax, len(pmf)), pmf / np.sum(pmf)])

    def kld1(self, x1, x2):
        def kli(a, b): return np.sum(np.multiply(a, np.log(a)) - np.multiply(a, np.log(b)), axis=0)
        x1 = 1.0 * x1 / np.sum(x1, axis=0)
        x2 = 1.0 * x2 / np.sum(x2, axis=0)
        return (kli(x1, x2) + kli(x2, x1)) / 2

    def kld2(self, x1, x2):
        from scipy.stats import entropy
        return entropy(x1, x2)

    def plot(self, tlim = 100):
        from pymc3.plots.kdeplot import fast_kde
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        est = []
        for rec in self.log: est.append([rec[0], rec[1][np.argmax(rec[2])], 0., rec[3][np.argmax(rec[4])], 0.])
        est = np.asarray(est)
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(est[:tlim,1], est[:tlim,0], est[:tlim,2], c='r', linewidth=1)
        ax.plot3D(est[:tlim,3], est[:tlim,0], est[:tlim,4], c='r', linewidth=1)
        for rec in self.log[:tlim]:
            ax.plot3D(rec[1], rec[0] * np.ones(len(rec[1])), rec[2], c='g', linewidth=1)
            ax.plot3D(rec[3], rec[0] * np.ones(len(rec[3])), rec[4], c='b', linewidth=1)
        plt.title('kl-divergence, %.1E' % (self.kld1(est[:tlim, 1], est[:tlim, 3])))
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('p')
        plt.show()

class Innovs():
    def __init__(self):
        self.log = []

    def update(self, rec):
        self.log.append(rec)

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

    def plot(self, x1=3):
        self.log = np.asarray(self.log[x1:])
        self.tts = self.log[:, 0]
        self.ets = self.log[:, 4]
        self.genzmw()
        lw = 1
        plt.subplot(3, 2, 1), plt.plot(self.tts, self.log[:, 1], linewidth=lw), plt.ylabel('xhat')
        plt.subplot(3, 2, 2), plt.plot(self.tts, self.log[:, 3], linewidth=lw), plt.ylabel('err')
        plt.subplot(3, 2, 3), plt.plot(self.tts, self.log[:, 2], linewidth=lw), plt.ylabel('yhat')
        plt.subplot(3, 2, 4), plt.plot(self.tts, self.log[:, 4], linewidth=lw), plt.ylabel('innov')
        plt.subplot(3, 2, 5), plt.plot(self.tts, self.zmw, linewidth=lw), plt.ylabel('autocorr')
        plt.show()
