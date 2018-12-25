import numpy as np
import math
from innovations import Innovations

class Modern():
    def __init__(self, mode, plot=True):
        self.log = []
        from models import Jazwinski1
        m = Jazwinski1()
        if mode == 'sigmapoint':
            self.jazwinski_sigmapoint(m)
        innov = Innovations(self.log)
        if plot: innov.plot_standard()

    def jazwinski_sigmapoint(self, m):
        xhat, Ptil = 2.2, .01
        W = np.array([m.kappa / float(m.bk), .5 / float(m.bk), .5 / float(m.bk)])
        for step in m.steps():
            X = m.va(m.X(xhat, Ptil), 0)
            Y = m.vc(m.Xhat(X, m.Rww), 0)
            Rksiksi = W @ np.power(Y - W @ Y, 2) + m.Rvv
            RXtilksi = W @ np.multiply(X - W @ X, Y - W @ Y)
            K = RXtilksi / Rksiksi
            xhat = W @ X + K * (step[2] - W @ Y)
            Ptil = W @ np.power(X - W @ X, 2) + m.Rww - K * Rksiksi * K
            self.log.append([step[0], W @ X, W @ Y, step[1] - W @ X, step[2] - W @ Y])

if __name__ == "__main__":
    Modern('sigmapoint')
