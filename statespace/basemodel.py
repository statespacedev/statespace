import numpy as np
import matplotlib.pyplot as plt


class BaseModel:
    '''base model'''

    def ekf(self): return None

    def ekfud(self): return None

    def sp(self): return None

    def spcho(self): return None

    def pf(self): return None

    def __init__(self):
        self.log = []

    def sim(self): pass

    def f(self, x): return None

    def F(self, x): return None

    def h(self, x): return None

    def H(self, x): return None


class SPKFBase():
    def __init__(self): pass

    def X1(self, xhat, Ptil): return None

    def X2(self, X): return None


class PFBase():
    def __init__(self): pass

    def X0(self): pass

    def predict(self, X): return None

    def update(self, X, o): return None

    def resample(self, X, W): return None


class EvalBase():
    def __init__(self):
        self.autocorr = Autocorr()  # override this in child's constructor with self.autocorr = Autocorr(parent)

    def model(self): return None

    def estimate(self, proclog): return None

    def show(self): plt.show()


class Log():
    def __init__(self, log):
        self.t = np.array([x[0] for x in log])
        self.x = np.array([x[1] for x in log])
        self.y = np.array([x[2] for x in log])


class Autocorr():
    def __init__(self, parent=None):
        self.parent = parent

    def run(self, proclog):
        logp = Log(proclog)
        logm = Log(self.parent.log)
        self.tts = logm.t
        self.ets = logm.y - logp.y
        self.mhate = np.mean(self.ets)
        self.Rhatee = np.mean(np.power(self.ets, 2))
        n = len(self.ets)
        self.tau = 1.96 * np.sqrt(self.Rhatee / n)
        self.iszeromean = True
        if abs(self.mhate) > self.tau: self.iszeromean = False
        self.zmw = self.autocorr1(self.ets)
        self.plot()

    def plot(self):
        plt.figure()
        lw = 1
        plt.plot(self.tts, self.zmw, linewidth=lw), plt.ylabel('autocorr')

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


if __name__ == "__main__":
    pass
