import math
import numpy as np

class Particle():
    '''particle filter, sequential monte carlo sampling processor, bootstrap particle filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []
        if 'pf1' in args: self.run = self.pf1
        else: self.run = self.pf2

    def pf1(self, model):
        '''particle filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        n, F, H = model.pf()
        X = x + math.sqrt(Q[0, 0]) * np.random.randn(n)
        W = self.normalize(np.ones((1, X.shape[1])))
        for t, o, u in sim():
            X = self.resample(X, W)
            X = F(X)
            W = self.normalize(H(o, X))
            self.log.append([t, W @ X.T, h(W @ X.T)])

    def pf2(self, model):
        '''particle filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        n, F, H = model.pf()
        x = x + np.multiply(np.random.randn(x.shape[0], n), np.diag(np.sqrt(Q)).reshape(-1, 1))
        W = np.ones((1, n)) / n
        for t, o, u in sim():
            x = np.row_stack((self.resample(x[0, :], W), self.roughen(x[1, :]), self.roughen(x[2, :])))
            x[0, :] = np.apply_along_axis(F, 0, x)
            W = self.normalize(H(o, x[0, :]))
            self.log.append([t, (W @ x.T).reshape(-1, 1), h((W @ x.T).reshape(-1, 1))])

    def resample(self, xi, Wi):
        '''particle resampling.'''
        tmp, xi = [], xi.reshape(1, -1)
        for i in range(xi.shape[1]):
            tmp.append([xi[0, i], Wi[0, i]])
        tmp = sorted(tmp, key=lambda x: x[0])
        cdf = [[tmp[0][0], tmp[0][1]]]
        for i in range(1, len(tmp)):
            cdf.append([tmp[i][0], tmp[i][1] + cdf[i - 1][1]])
        cdf = np.asarray(cdf)
        uk = np.sort(np.random.uniform(size=xi.shape[0]))
        xhati, k = [], 0
        for row in cdf:
            if k < uk.size and uk[k] <= row[1]:
                xhati.append(row[0])
                k += 1
            else: xhati.append(cdf[0][0])
        xhati = np.asarray(xhati).reshape(1, -1)
        return xhati

    def roughen(self, x):
        x = x.reshape(1, -1)
        tmp = x + .1 * np.random.randn(1, x.shape[1])
        return tmp

    def normalize(self, W):
        return W / np.sum(W)

if __name__ == "__main__":
    pass
