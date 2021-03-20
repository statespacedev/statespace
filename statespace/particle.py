import numpy as np
import math, util
import sys; sys.path.append('../')
from models.threestate import Threestate
from models.onestate import Onestate

def main():
    run('pf1')
    run('pf2')

def run(mode='pf1'):
    processor = Particle()
    if mode == 'pf1':
        processor.pf1(Onestate())
        # processor.dists.plot()
    if mode == 'pf2': processor.pf2(Threestate())
    if mode == 'ensemble':
        for i in range(100):
            processor.pf1(Onestate())
            # processor.ens.update()
        processor.ens.plot()
    processor.innov.plot()

class Particle():
    '''particle filter. monte carlo sampling processor, bootstrap particle filter. the run methods bring in particular models from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods.'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.innov = util.Innovs()
        self.dists = util.Dists()
        self.ens = util.DistsEns()

    def pf1(self, model):
        '''particle filter 1.'''
        xhat = 2.05
        x = xhat + math.sqrt(1e-2) * np.random.randn(model.nsamp)
        W = self.normalize(np.ones(model.nsamp))
        for step in model.steps():
            x = self.resample(x, W)
            x = model.Apf(x)
            W = self.normalize(model.Cpf(step[2], x))
            yhat = model.c(xhat, 0)
            xhat = W @ x
            self.innov.update(t=step[0], xhat=xhat, yhat=yhat, err=step[1] - xhat, inn=step[2] - yhat)
            self.dists.update(t=step[0], tru=model.Apf(step[1]), est=x)

    def pf2(self, model):
        '''particle filter 2.'''
        xhat = np.array([2.0, .055, .044])
        x = xhat + np.sqrt(model.Rww) * np.random.randn(model.nsamp, 3)
        W = np.ones((model.nsamp, 3)) / model.nsamp
        for step in model.steps():
            x[:, 0], x[:, 1], x[:, 2] = self.resample(x[:, 0], W[:, 0]), self.roughen(x[:, 1]), self.roughen(x[:, 2])
            x[:, 0] = np.apply_along_axis(model.Apf, 1, x)
            W[:, 0] = self.normalize(model.Cpf(step[2], x[:, 0]))
            yhat = model.c(xhat, 0)
            xhat = [W[:, 0].T @ x[:, 0], W[:, 1].T @ x[:, 1], W[:, 2].T @ x[:, 2]]
            self.innov.update(t=step[0], xhat=xhat[0], yhat=yhat, err=step[1][0] - xhat[0], inn=step[2] - yhat)

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
