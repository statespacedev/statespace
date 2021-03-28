import math
import numpy as np

class Particle():
    '''particle filter, sequential monte carlo sampling processor, bootstrap particle filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []

    def pf1(self, model): # todo still scalar
        '''particle filter'''
        sim, f, h, F, H, R, x, P = model.ekf()
        nsamp, F, H = model.pf()
        x = x + math.sqrt(1e-2) * np.random.randn(nsamp)
        W = self.normalize(np.ones(nsamp))
        for t, obs in sim():
            x = self.resample(x, W)
            x = F(x)
            y = h(x)
            W = self.normalize(H(obs, x))
            x = W @ x
            self.log.append([t, x, y])

    def pf2(self, model):
        '''particle filter'''
        sim, f, h, F, H, R, x, P = model.ekf()
        nsamp, F, H = model.pf()
        x = x + np.sqrt(model.Rww) * np.random.randn(model.PF.nsamp, len(x))
        W = np.ones((nsamp, 3)) / nsamp
        for t, obs in sim():
            x[:, 0], x[:, 1], x[:, 2] = self.resample(x[:, 0], W[:, 0]), self.roughen(x[:, 1]), self.roughen(x[:, 2])
            x[:, 0] = np.apply_along_axis(F, 1, x)
            y = model.h(x)
            W[:, 0] = self.normalize(H(obs, x[:, 0]))
            x = [W[:, 0].T @ x[:, 0], W[:, 1].T @ x[:, 1], W[:, 2].T @ x[:, 2]]
            self.log.append([t, x, y])

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
    pass
