# monte carlo sampling processor, bootstrap particle filter
import numpy as np
import math
from innovations import Innovations

def resample(xi, Wi):
    tmp = []
    for i in range(xi.size):
        tmp.append([xi[i], Wi[i]])
    tmp = sorted(tmp, key=lambda x: x[0])
    cdf = [[tmp[0][0], tmp[0][1]]]
    for i in range(1, len(tmp)):
        cdf.append([tmp[i][0], tmp[i][1] + cdf[i - 1][1]])
    cdf = np.asarray(cdf)
    uk = np.sort(np.random.uniform(size=xi.size))
    xhati, k = [], 0
    for row in cdf:
        while k < uk.size and uk[k] <= row[1]:
            xhati.append(row[0])
            k += 1
    xhati = np.asarray(xhati)
    assert xhati.size == xi.size
    return xhati

class Particle():
    def __init__(self, mode, plot=True):
        self.log = []
        from models import Jazwinski1
        m = Jazwinski1()
        if mode == 'bootstrap':
            self.jazwinski_bootstrap(m)
        innov = Innovations(self.log)
        if plot: innov.plot_standard()

    def jazwinski_bootstrap(self, m):
        xhat = 2.05
        wi = math.sqrt(m.Rww) * np.random.randn(m.nsamp)
        xi = xhat + wi
        for step in m.steps():
            wi = math.sqrt(m.Rww) * np.random.randn(m.nsamp)
            xi = m.vAcurl(xi, wi)
            Wi = m.vCcurl(step[2], xi)
            Wi = Wi / sum(Wi)
            xhat = Wi @ xi
            yhat = m.c(xhat, 0)
            xi = resample(xi, Wi)
            self.log.append([step[0], xhat, yhat, step[1] - xhat, step[2] - yhat])

if __name__ == "__main__":
    Particle('bootstrap')
