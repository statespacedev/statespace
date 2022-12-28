"""placeholder for what could grow to become a higher-level statespace model - with individual models inheriting and
overriding. for now it's mostly acting as a skelton to sketch out what such a beast could potentially look like -
what entities and activities make sense here?

if a particular model doesn't need something sketched out here, don't override it. if something's needed,
override it.
"""
import numpy as np
import matplotlib.pyplot as plt


class Model:
    """what's common across our models? """

    def __init__(self, conf):
        self.conf = conf
        self.log = []

    def ekf(self):
        """entities needed for an extended kalman filter processor. """
        return None

    def spkf(self):
        """entities needed for a sigma point kalman filter processor. """
        return None

    def spkf_cholesky(self):
        """entities need for cholesky factorization sigma point kalman filter processor. """
        return None

    def pf(self):
        """entities needed for a particle filter processor. """
        return None

    def sim(self):
        """simulation states. a time series of true states and obs. """
        pass

    def f(self, x):
        """state evolution equation. """
        return None

    def F(self, x):
        """state evolution matrix. """
        return None

    def h(self, x):
        """observation equation. """
        return None

    def H(self, x):
        """observation sensitivity matrix. """
        return None


# noinspection PyUnusedLocal
class SPKFCommon:
    """what's common across sigma point kalman filter models? """

    def __init__(self):
        pass

    def XY(self, x, P, u):
        """sigma points update. """
        return None

    def XYcho(self, X, S, u):
        """cholesky sigma points update. """
        return None


class PFCommon:
    def __init__(self): pass

    def X0(self):
        """initial particles. """
        pass

    def predict(self, X, u):
        """evolution forward in time of particles. """
        return None

    def update(self, X, o):
        """observational update of particles. """
        return None

    def resample(self, X, W):
        """resampling of particles. """
        return None


class EvalCommon:
    """evaluating processor results. """

    def __init__(self):
        self.autocorr = Autocorr()  # override this in child's constructor with self.autocorr = Autocorr(parent)

    def model(self):
        """plot true timeseries. """
        return None

    def estimate(self, proclog):
        """plot estimated results. """
        return None

    @staticmethod
    def show():
        """show the plots. """
        plt.show()


class Log:
    """logging of the results timeseries. """

    def __init__(self, log):
        self.t = np.array([x[0] for x in log])
        self.x = np.array([x[1] for x in log])
        self.y = np.array([x[2] for x in log])


class Autocorr:
    """auto correlation of results with themselves. """

    def __init__(self, parent=None):
        self.parent = parent

    def run(self, proclog):
        """run autocorrelation. """
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
        """show the plot. """
        plt.figure()
        lw = 1
        plt.plot(self.tts, self.zmw, linewidth=lw), plt.ylabel('autocorr')

    def autocorr1(self, x):
        """calculation version one. """
        n = x.size
        ac = np.zeros([n, 1])
        for k in range(1, n):  # k = 1 to n-1
            ac[k] = 0
            for t in range(1, n - k):  # t = 1 to n-k
                ac[k] += (self.ets[t - 1] - self.mhate) * (self.ets[t - 1 + k] - self.mhate)  # / (n-k)
        return ac

    @staticmethod
    def autocorr2(x):
        """calculation version two. """
        ac = np.correlate(x, x, mode='full')
        return ac[ac.size // 2:]


if __name__ == "__main__":
    pass
