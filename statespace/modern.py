import numpy as np
from innovations import Innovations
import models

class Modern():
    def __init__(self, mode, plot=True):
        self.log = []
        if mode == 'spkf1':
            m = models.Jazwinski1()
            self.spkf1(m)
        elif mode == 'spkf2':
            m = models.Jazwinski2()
            self.spkf2(m)
        innov = Innovations(self.log)
        if plot: innov.plot_standard()

    def spkf1(self, m):
        xhat, Ptil = 2.2, .01
        W = np.array([m.k1, m.k2, m.k2])
        for step in m.steps():
            X = m.va(m.X(xhat, Ptil))
            Y = m.vc(m.Xhat(X, m.Rww))
            Rksiksi = W @ np.power(Y - W @ Y, 2) + m.Rvv
            RXtilksi = W @ np.multiply(X - W @ X, Y - W @ Y)
            K = RXtilksi / Rksiksi
            yhat = W @ Y
            xhat = W @ X + K * (step[2] - W @ Y)
            Ptil = W @ np.power(X - W @ X, 2) + m.Rww - K * Rksiksi * K
            self.log.append([step[0], xhat, yhat, step[1] - xhat, step[2] - yhat])

    def spkf2(self, m):
        xhat = np.array([2.0, .055, .044])
        Ptil = .1 * np.eye(3)
        W = np.array([m.k1, m.k2, m.k2, m.k2, m.k2, m.k2, m.k2])
        for step in m.steps():
            X = m.va(m.X(xhat, Ptil))
            Y = m.vc(m.Xhat(X, m.Rww))
            Rksiksi = W * m.ksi(Y, W) @ m.ksi(Y, W).T + m.Rvv
            RXtilksi = W * m.Xtil(X, W) @ m.ksi(Y, W).T
            K = np.squeeze(np.asarray(RXtilksi / Rksiksi))
            yhat = W @ Y
            xhat = W @ X.T + K * (step[2] - W @ Y)
            Ptil = Ptil # W * m.Xtil(X, W) @ m.Xtil(X, W).T + m.Rww - K @ K.T * Rksiksi
            self.log.append([step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat])

if __name__ == "__main__":
    Modern('spkf1')
    Modern('spkf2')

