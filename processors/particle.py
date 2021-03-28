import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import math, util
import sys; sys.path.append('../')
from models.threestate import Threestate
from models.onestate import Onestate
from innovations import Innovs

def main():
    processor = Particle()
    # model = Onestate()
    model = Threestate()
    # processor.pf1(model)
    processor.pf2(model)
    processor.innovs.plot()

class Particle():
    '''particle filter, sequential monte carlo sampling processor, bootstrap particle filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.innovs = args, kwargs, Innovs()

    def pf1(self, model): # todo still scalar
        '''particle filter'''
        steps, f, h, F, H, R, x, P = model.ekf()
        nsamp, F, H = model.pf()
        xp = x + math.sqrt(1e-2) * np.random.randn(nsamp)
        W = self.normalize(np.ones(nsamp))
        for t, xt, yt in steps():
            xp = self.resample(xp, W)
            xp = F(xp)
            y = h(x)
            W = self.normalize(H(yt, xp))
            x = W @ xp
            self.innovs.add(t, xt, yt, x, y)

    def pf2(self, model):
        '''particle filter'''
        steps, f, h, F, H, R, x, P = model.ekf()
        nsamp, F, H = model.pf()
        xp = x + np.sqrt(model.Rww) * np.random.randn(model.PF.nsamp, len(x))
        W = np.ones((nsamp, 3)) / nsamp
        for t, xt, yt in steps():
            xp[:, 0], xp[:, 1], xp[:, 2] = self.resample(xp[:, 0], W[:, 0]), self.roughen(xp[:, 1]), self.roughen(xp[:, 2])
            xp[:, 0] = np.apply_along_axis(F, 1, xp)
            y = model.h(x)
            W[:, 0] = self.normalize(H(yt, xp[:, 0]))
            x = [W[:, 0].T @ xp[:, 0], W[:, 1].T @ xp[:, 1], W[:, 2].T @ xp[:, 2]]
            self.innovs.add(t, xt, yt, x, y)

    def resample(self, xi, Wi):
        '''particle resampling.'''
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

    def roughen(self, x):
        return x + .1 * np.random.randn(x.size)

    def normalize(self, W):
        return W / sum(W)

if __name__ == "__main__":
    main()
