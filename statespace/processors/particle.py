import math
import numpy as np


class Particle:
    '''particle filter, sequential monte carlo processor'''

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []
        self.run = self.pf

    def pf(self, model):
        '''basic sampling-importance-resampling (SIR) particle filter, bootstrap particle filter, condensation particle filter, survival of the fittest algorithm'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        X, predict, update, resample = model.pf()
        for t, o, u in sim():
            X = predict(X, u)
            W = update(X, o);
            x = np.sum(np.multiply(W, X), axis=1).reshape(-1, 1)
            X = resample(X, W)
            self.log.append([t, x, h(x)])


if __name__ == "__main__":
    pass
