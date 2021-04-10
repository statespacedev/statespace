import math
import numpy as np

class Particle():
    '''particle filter, sequential monte carlo sampling processor'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []
        self.run = self.bootstrap

    def bootstrap(self, model):
        '''basic sampling-importance-resampling (SIR) particle filter, bootstrap particle filter, condensation particle filter, survival of the fittest algorithm'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        X, predict, update, resample = model.pf()
        for t, o, u in sim():
            X = predict(X)
            W = update(X, o)
            X = resample(X, W)
            self.log.append([t, W @ X.T, h(W @ X.T)])

if __name__ == "__main__":
    pass
