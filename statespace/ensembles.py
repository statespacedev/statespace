import math
import numpy as np
import matplotlib.pyplot as plt

class DecisionEnsemble():
    def __init__(self):
        pass

class DistsEnsemble():
    def __init__(self):
        self.log = []

    def update(self, distslog):
        for rec in distslog:
            truval, trupmf = rec[1], rec[2]
            maptru = truval[np.argmax(trupmf)]
            estval, estpmf = rec[3], rec[4]
            mapest = estval[np.argmax(estpmf)]
            self.log.append([maptru, mapest])

    def plot(self):
        log = np.asarray(self.log)
        from pymc3.plots.kdeplot import fast_kde
        import matplotlib.pyplot as plt
        maptrupmf, maptrumin, maptrumax = fast_kde(log[:, 0])
        mapestpmf, mapestmin, mapestmax = fast_kde(log[:, 1])
        maptruval = np.linspace(maptrumin, maptrumax, len(maptrupmf))
        mapestval = np.linspace(mapestmin, mapestmax, len(mapestpmf))
        maptrupmf = maptrupmf / sum(mapestpmf)
        mapestpmf = mapestpmf / sum(mapestpmf)
        plt.figure()
        plt.plot(maptruval, maptrupmf, 'g')
        plt.plot(mapestval, mapestpmf, 'b')
        plt.show()
        pass

class ModelEns():
    def __init__(self, title):
        self.title = title
        self.runs = []

    def runningmean(self, x, n):
        ypadded = np.pad(x, (n // 2, n - 1 - n // 2), mode='edge')
        return np.convolve(ypadded, np.ones((n,)) / n, mode='valid')

    def finalize(self):
        self.ensx = []
        self.ensy = []
        for log in self.runs:
            log = np.asarray(log)
            self.ensx.append(log[:, 1])
            self.ensy.append(log[:, 2])
        self.ensx = np.asarray(self.ensx)
        self.ensy = np.asarray(self.ensy)
        self.ensxmean = self.runningmean(np.mean(self.ensx, axis=0), 20)
        self.ensymean = self.runningmean(np.mean(self.ensy, axis=0), 20)
        self.ensxstd = self.runningmean(np.std(self.ensx, axis=0), 20)
        self.ensystd = self.runningmean(np.std(self.ensy, axis=0), 20)
        pass

    def plot(self):
        self.finalize()
        lw = 1
        plt.figure()
        plt.subplot(2, 1, 1)
        for log in self.runs:
            log = np.asarray(log)
            plt.plot(log[:, 0], log[:, 1], linewidth=lw, color='g', alpha=.25)
            plt.plot(log[:, 0], self.ensxmean, linewidth=lw, color='b', alpha=.1)
            plt.plot(log[:, 0], self.ensxmean + 1.96 * self.ensxstd, linewidth=lw, color='b', alpha=.05)
            plt.plot(log[:, 0], self.ensxmean - 1.96 * self.ensxstd, linewidth=lw, color='b', alpha=.05)
            plt.ylabel('x')
            if self.title: plt.title(self.title)
        plt.subplot(2, 1, 2)
        for log in self.runs:
            log = np.asarray(log)
            plt.plot(log[:, 0], log[:, 2], linewidth=lw, color='g', alpha=.25)
            plt.plot(log[:, 0], self.ensymean, linewidth=lw, color='b', alpha=.1)
            plt.plot(log[:, 0], self.ensymean + 1.96 * self.ensystd, linewidth=lw, color='b', alpha=.05)
            plt.plot(log[:, 0], self.ensymean - 1.96 * self.ensystd, linewidth=lw, color='b', alpha=.05)
            plt.ylabel('y')

