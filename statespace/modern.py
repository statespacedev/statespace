import numpy as np
import math
from innovations import Innovations

class Modern():
    def __init__(self, mode, plot=True):
        self.log = []
        if mode == 'sigmapoint':
            from models import Jazwinski1
            m = Jazwinski1()
            self.jazwinski_sigmapoint(m)
        elif mode == 'sigmapoint2':
            from models import Jazwinski2
            m = Jazwinski2()
            self.jazwinski_sigmapoint2(m)
        innov = Innovations(self.log)
        if plot: innov.plot_standard()

    def jazwinski_sigmapoint(self, m):
        xhat, Ptil = 2.2, .01
        W = np.array([m.k1, m.k2, m.k2])
        for step in m.steps():
            X = m.va(m.X(xhat, Ptil), 0)
            Y = m.vc(m.Xhat(X, m.Rww), 0)
            Rksiksi = W @ np.power(Y - W @ Y, 2) + m.Rvv
            RXtilksi = W @ np.multiply(X - W @ X, Y - W @ Y)
            K = RXtilksi / Rksiksi
            xhat = W @ X + K * (step[2] - W @ Y)
            Ptil = W @ np.power(X - W @ X, 2) + m.Rww - K * Rksiksi * K
            self.log.append([step[0], W @ X, W @ Y, step[1] - W @ X, step[2] - W @ Y])

    def jazwinski_sigmapoint2(self, m):
        xhat = np.array([2.2, .055, .044])
        Ptil = 1. * np.eye(3)
        W = np.array([m.k1, m.k2, m.k2, m.k2, m.k2, m.k2, m.k2])
        for step in m.steps():
            X = m.va(m.X(xhat, Ptil))
            Y = m.vc(m.Xhat(X, m.Rww))
            Rksiksi = (Y - W @ Y) @ np.diag(W) @ (Y - W @ Y).T + m.Rvv
            RXtilksi = (X.T - W @ X.T).T @ np.diag(W) @ (Y - W @ Y)
            K = RXtilksi / Rksiksi
            xhat = W @ X.T + K * (step[2] - W @ Y)
            Ptil = Ptil - K @ (Rksiksi * K.T)
            self.log.append([step[0], (W @ X.T)[0], W @ Y, step[1][0] - (W @ X.T)[0], step[2] - W @ Y])

if __name__ == "__main__":
    # Modern('sigmapoint')
    Modern('sigmapoint2')

