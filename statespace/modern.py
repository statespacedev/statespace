# sigma-point bayesian processor, unscented kalman filter, SPBP, UKF
import numpy as np
import math
from innov import Innov

class Modern():
    def __init__(self):
        pass

    def spbp1(self):
        n = 150
        deltat = .01

        xhat0 = 2.2
        Ptil0 = .01

        bignsubx = 1
        kappa = 1
        bk = bignsubx + kappa
        W = np.zeros((3,))
        W[:] = [kappa / float(bk), .5 / float(bk), .5 / float(bk)]

        def fx(x, w):
            return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

        def fy(x, v):
            return x ** 2 + x ** 3 + v

        vfx = np.vectorize(fx)
        vfy = np.vectorize(fy)

        def vfa(vx):
            return vfx(vx, 0)

        def vfc(vx):
            return vfy(vx, 0)

        def vfX(xhat, Ptil):
            return [xhat, xhat + math.sqrt(bk * Ptil), xhat - math.sqrt(bk * Ptil)]

        def Xhat(X, Rww):
            return [X[0], X[1] + kappa * math.sqrt(Rww), X[2] - kappa * math.sqrt(Rww)]

        tts = np.arange(0, n * deltat, deltat)
        Rvv = .09
        vts = math.sqrt(Rvv) * np.random.randn(n)
        Rww = 0
        wts = math.sqrt(Rww) * np.random.randn(n)

        xts = np.zeros((n,))
        xts[0] = 2.
        yts = np.zeros((n,))
        yts[0] = fy(xts[0], vts[0])

        Xts = np.zeros((n, 3))
        Xts[0, :] = vfX(xhat0, Ptil0)
        xhatts = np.zeros((n,))
        xhatts[0] = xhat0
        xtilts = np.zeros((n,))
        xtilts[0] = xts[0] - xhatts[0]

        Xtilts = np.zeros((n, 3))
        Xtilts[0, :] = Xts[0, :] - xhatts[0]
        Ptilts = np.zeros((n,))
        Ptilts[0] = Ptil0

        Xhatts = np.zeros((n, 3))
        Xhatts[0, :] = Xts[0, :]

        Yts = np.zeros((n, 3))
        Yts[0, :] = vfc(Xhatts[0, :])
        yhatts = np.zeros((n,))
        yhatts[0] = W @ Yts[0, :]
        ets = np.zeros((n,))
        ets[0] = yts[0] - yhatts[0]

        ksits = np.zeros((n, 3))
        ksits[0, :] = Yts[0, :] - yhatts[0]
        Rksiksits = np.zeros((n,))
        Rksiksits[0] = W @ np.power(ksits[0, :], 2) + Rvv

        RXtilksits = np.zeros((n,))
        RXtilksits[0] = W @ np.multiply(Xtilts[0, :], ksits[0, :])
        Kts = np.zeros((n,))
        Kts[0] = RXtilksits[0] / Rksiksits[0]

        for tk in range(1, n):
            xts[tk] = fx(xts[tk - 1], wts[tk - 1])
            yts[tk] = fy(xts[tk], vts[tk])

            X = vfX(xhatts[tk-1], Ptilts[tk-1])
            Xts[tk, :] = vfa(X)
            xhatts[tk] = W @ Xts[tk, :]
            Xtilts[tk, :] = Xts[tk, :] - xhatts[tk]
            Ptilts[tk] = W @ np.power(Xtilts[tk, :], 2) + Rww

            Xhatts[tk, :] = Xhat(Xts[tk, :], Rww)

            Yts[tk, :] = vfc(Xhatts[tk, :])
            yhatts[tk] = W @ Yts[tk, :]
            ets[tk] = yts[tk] - yhatts[tk]

            ksits[tk, :] = Yts[tk, :] - yhatts[tk]
            Rksiksits[tk] = W @ np.power(ksits[tk, :], 2) + Rvv

            RXtilksits[tk] = W @ np.multiply(Xtilts[tk, :], ksits[tk, :])
            Kts[tk] = RXtilksits[tk] / Rksiksits[tk]

            xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
            Ptilts[tk] = Ptilts[tk] - Kts[tk] * Rksiksits[tk] * Kts[tk]
            xtilts[tk] = xts[tk] - xhatts[tk]

        innov = Innov(tts, ets)
        innov.standard(tts, xhatts, xtilts, yhatts)

    def spbp2(self):
        n = 150
        deltat = .01
        tts = np.arange(0, n * deltat, deltat)

        Rww = np.diag([0, 0, 0])
        wts = np.multiply(np.random.randn(n, 3), np.sqrt(np.diag(Rww)))
        xts = np.zeros([n, 3])
        xts[0, :] = [2., .05, .04]

        def fx(x, w):
            return np.array([(1 - x[1] * deltat) * x[0] + x[2] * deltat * x[0] ** 2, x[1], x[2]]) + w

        for tk in range(1, n):
            xts[tk, :] = fx(xts[tk - 1, :], wts[tk - 1, :])

        Rvv = .09
        vts = math.sqrt(Rvv) * np.random.randn(n)
        yts = np.zeros(n)

        def fy(x, v):
            return x[0] ** 2 + x[0] ** 3 + v

        yts[0] = fy(xts[0, :], vts[0])
        for tk in range(1, n):
            yts[tk] = fy(xts[tk, :], vts[tk])

        nx, kappa = 3, 1
        a, b = kappa / float(nx + kappa), 1 / float(2 * (nx + kappa))
        W = np.array([a, b, b, b, b, b, b])

        tk = 0

        def fX(x, P):
            X = np.zeros([3, 7])
            X[:, 0] = x
            X[:, 1] = x + np.array([math.sqrt((nx + kappa) * P[0, 0]), 0, 0])
            X[:, 2] = x + np.array([0, math.sqrt((nx + kappa) * P[1, 1]), 0])
            X[:, 3] = x + np.array([0, 0, math.sqrt((nx + kappa) * P[2, 2])])
            X[:, 4] = x - np.array([math.sqrt((nx + kappa) * P[0, 0]), 0, 0])
            X[:, 5] = x - np.array([0, math.sqrt((nx + kappa) * P[1, 1]), 0])
            X[:, 6] = x - np.array([0, 0, math.sqrt((nx + kappa) * P[2, 2])])
            return X

        def fa(X):
            for i in range(7):
                X[:, i] = fx(X[:, i], 0)
            return X

        xhat = [2., .055, .044]
        Ptil = .01 * np.eye(3)

        def fXhat(X, Rww):
            Xhat = np.zeros([3, 7])
            Xhat[:, 0] = X[:, 0]
            Xhat[:, 1] = X[:, 1] + np.array([kappa * math.sqrt(Rww[0, 0]), 0, 0])
            Xhat[:, 2] = X[:, 2] + np.array([0, kappa * math.sqrt(Rww[1, 1]), 0])
            Xhat[:, 3] = X[:, 3] + np.array([0, 0, kappa * math.sqrt(Rww[2, 2])])
            Xhat[:, 4] = X[:, 4] - np.array([kappa * math.sqrt(Rww[0, 0]), 0, 0])
            Xhat[:, 5] = X[:, 5] - np.array([0, kappa * math.sqrt(Rww[1, 1]), 0])
            Xhat[:, 6] = X[:, 6] - np.array([0, 0, kappa * math.sqrt(Rww[2, 2])])
            return Xhat

        def fc(Xhat):
            Y = np.zeros(7)
            for i in range(7):
                Y[i] = fy(Xhat[:, i], 0)
            return Y

        X = fX(xhat, Ptil)
        Xhat = fXhat(X, Rww)
        Y = fc(Xhat)
        yhat = W @ Y
        e = yts[tk] - yhat
        ep = Y - yhat
        Repep = ep @ np.diag(W) @ ep.T + Rvv

        xhatts = np.zeros([n, 3])
        xtilts = np.zeros([n, 3])
        yhatts = np.zeros(n)
        ets = np.zeros(n)
        for tk in range(1, len(tts)):
            X = fa(fX(xhat, Ptil))
            xhat = W @ X.T
            Xtil = (X.T - xhat).T
            Ptil = Xtil @ np.diag(W) @ Xtil.T + Rww
            X = fX(xhat, Ptil)
            Xhat = fXhat(X, Rww)
            Y = fc(Xhat)
            yhat = W @ Y
            e = yts[tk] - yhat
            ep = Y - yhat
            Repep = ep @ np.diag(W) @ ep.T + Rvv
            Rxtilep = Xtil @ np.diag(W) @ ep
            K = Rxtilep / Repep
            # xhat = xhat + K * e
            # Ptil = Ptil - K @ (Repep * K.T)

            xhatts[tk, :] = xhat
            xtilts[tk, :] = xts[tk] - xhat
            yhatts[tk] = yhat
            ets[tk] = e
        innov = Innov(tts, ets)
        innov.abp(tts, xhatts, xtilts, yhatts)

if __name__ == "__main__":
    mod = Modern()
    mod.spbp1()
    mod.spbp2()
