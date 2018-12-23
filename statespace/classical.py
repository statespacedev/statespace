import numpy as np
import util, math
from innov import Innov

class Classical():
    def __init__(self):
        pass

    def lzbp(self):
        n = 150
        deltat = .01

        def x(x, w):
            return (1 - .05 * deltat) * x + .04 * deltat * x ** 2 + w

        def y(x, v):
            return x ** 2 + x ** 3 + v

        def A(x):
            return 1 - .05 * deltat + .08 * deltat * x

        def C(x):
            return 2 * x + 3 * x ** 2

        tts = np.arange(0, n * deltat, deltat)
        Rww = 0
        wts = math.sqrt(Rww) * np.random.randn(n)
        Rvv = .09
        vts = math.sqrt(Rvv) * np.random.randn(n)
        xts = np.zeros_like(tts)
        xts[0] = 2.
        yts = np.zeros_like(tts)
        yts[0] = y(xts[0], vts[0])
        xrefts = np.zeros_like(tts)
        xrefts[0] = 2.

        xhatts = np.zeros_like(tts)
        xhatts[0] = 2.2
        Ptilts = np.zeros_like(tts)
        Ptilts[0] = .01
        xtilts = np.zeros_like(tts)
        xtilts[0] = xts[0] - xhatts[0]

        Reets = np.zeros_like(tts)
        Reets[0] = 0
        Kts = np.zeros_like(tts)
        Kts[0] = 0

        yhatts = np.zeros_like(tts)
        yhatts[0] = y(xrefts[0], 0) + C(xrefts[0]) * (xhatts[0] - xrefts[0])
        ets = np.zeros_like(tts)
        ets[0] = yts[0] - yhatts[0]

        for tk in range(1, n):
            xts[tk] = x(xts[tk - 1], wts[tk - 1])
            yts[tk] = y(xts[tk], vts[tk])
            xrefts[tk] = 2. + .067 * tts[tk]

            xhatts[tk] = x(xrefts[tk - 1], 0) + A(xrefts[tk - 1]) * (xhatts[tk - 1] - xrefts[tk - 1])
            Ptilts[tk] = A(xrefts[tk - 1]) ** 2 * Ptilts[tk - 1]

            Reets[tk] = C(xhatts[tk]) ** 2 * Ptilts[tk] + Rvv
            Kts[tk] = Classical.div0(Ptilts[tk] * C(xrefts[tk]), Reets[tk])

            yhatts[tk] = y(xrefts[tk], 0) + C(xrefts[tk]) * (xhatts[tk] - xrefts[tk])
            ets[tk] = yts[tk] - yhatts[tk]

            xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
            Ptilts[tk] = (1 - Kts[tk] * C(xrefts[tk])) * Ptilts[tk]
            xtilts[tk] = xts[tk] - xhatts[tk]

        innov = Innov(tts, ets)
        innov.standard(tts, xhatts, xtilts, yhatts)

    def xbp1(self):
        n = 150
        deltat = .01
        tts = np.arange(0, n * deltat, deltat)

        Rww = 0
        wts = math.sqrt(Rww) * np.random.randn(n)
        xts = np.zeros(n)
        xts[0] = 2.
        def fx(x, w):
            return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w
        for tk in range(1, n):
            xts[tk] = fx(xts[tk - 1], wts[tk - 1])

        Rvv = .09
        vts = math.sqrt(Rvv) * np.random.randn(n)
        yts = np.zeros(n)
        def fy(x, v):
            return x**2 + x**3 + v
        yts[0] = fy(xts[0], vts[0])
        for tk in range(1, n):
            yts[tk] = fy(xts[tk], vts[tk])

        xhatts = np.zeros(n)
        xhatts[0] = 2.2
        def fA(x):
            return 1 - .05 * deltat + .08 * deltat * x
        Ptilts = np.zeros(n)
        Ptilts[0] = .01
        xtilts = np.zeros(n)
        xtilts[0] = xts[0] - xhatts[0]

        tk = 0
        yhatts = np.zeros(n)
        yhatts[tk] = fy(xhatts[tk], 0)
        ets = np.zeros(n)
        ets[tk] = yts[tk] - yhatts[tk]
        def fC(x):
            return 2 * x + 3 * x**2
        C = fC(xhatts[tk])
        Reets = np.zeros(n)
        Reets[tk] = C * Ptilts[tk] * C + Rvv

        for tk in range(1, n):
            Phi = fA(fx(xhatts[tk-1], 0))
            xhatts[tk] = fx(xhatts[tk-1], 0)
            Ptilts[tk] = Phi * Ptilts[tk-1] * Phi
            yhatts[tk] = fy(xhatts[tk], 0)
            ets[tk] = yts[tk] - yhatts[tk]
            C = fC(xhatts[tk])
            Reets[tk] = C * Ptilts[tk] * C + Rvv
            K = Ptilts[tk] * C / Reets[tk]
            xhatts[tk] = xhatts[tk] + K * ets[tk]
            Ptilts[tk] = (1 - K * C) * Ptilts[tk]
            xtilts[tk] = xts[tk] - xhatts[tk]

        innov = Innov(tts, ets)
        innov.standard(tts, xhatts, xtilts, yhatts)

    def xbp2(self):
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

        tk = 0
        xhatts = np.zeros([n, 3])
        Ptilts = np.zeros([n, 3, 3])
        yhatts = np.zeros(n)
        ets = np.zeros(n)
        Reets = np.zeros(n)
        xtilts = np.zeros([n, 3])

        def fA(x):
            A = np.eye(3)
            A[0, 0] = 1 - x[1] * deltat + 2 * x[2] * deltat * x[0]
            A[0, 1] = -deltat * x[0]
            A[0, 2] = deltat * x[0] ** 2
            return A

        xhatts[tk, :] = [2., .055, .044]
        Ptilts[tk, :, :] = 100. * np.eye(3)
        U, D = Classical.UD(Ptilts[tk, :, :])
        for tmp in range(1, 2):
            Phi = fA(fx(xhatts[tmp - 1, :], 0))
            xhat = Phi @ xhatts[tmp - 1, :]
            Ptil = Phi @ Ptilts[tmp - 1, :, :] @ Phi.T + Rww
            pass

        def fC(x):
            return np.array([2 * x[0] + 3 * x[0] ** 2, 0, 0])

        yhatts[tk] = fy(xhatts[tk, :], 0)
        ets[tk] = yts[tk] - yhatts[tk]
        C = fC(xhatts[tk, :])
        Reets[tk] = C @ Ptilts[tk, :, :] @ C + Rvv
        xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

        mode = 1
        for tk in range(1, n):
            Phi = fA(fx(xhatts[tk - 1, :], 0))
            if mode == 0:
                xhatts[tk, :] = Phi @ xhatts[tk - 1, :]
                Ptilts[tk, :, :] = Phi @ Ptilts[tk - 1, :, :] @ Phi.T + Rww
                yhatts[tk] = fy(xhatts[tk, :], 0)
                ets[tk] = yts[tk] - yhatts[tk]
                C = fC(xhatts[tk, :])
                Reets[tk] = C @ Ptilts[tk, :, :] @ C.T + Rvv
                K = Ptilts[tk, :, :] @ C.T / Reets[tk]
                xhatts[tk, :] = xhatts[tk, :] + K * ets[tk]
                Ptilts[tk, :, :] = (np.eye(3) - K @ C) @ Ptilts[tk, :, :]
            elif mode == 1:
                x, U, D = Classical.thornton(xin=xhatts[tk - 1, :], Phi=Phi, Uin=U, Din=D, Gin=np.eye(3), Q=Rww)
                yhatts[tk] = fy(x, 0)
                ets[tk] = yts[tk] - yhatts[tk]
                x, U, D = Classical.bierman(z=ets[tk], R=Rvv, H=fC(x), xin=x, Uin=U, Din=D)
                xhatts[tk, :] = x
            xtilts[tk, :] = xts[tk, :] - xhatts[tk, :]

        innov = Innov(tts, ets)
        innov.abp(tts, xhatts, xtilts, yhatts)

    @staticmethod
    def UD(M):
        assert np.allclose(M, M.T)
        n = M.shape[0]
        M = np.triu(M)
        U = np.eye(n)
        d = np.zeros(n)
        for j in reversed(range(2, n + 1)):
            d[j - 1] = M[j - 1, j - 1]
            if d[j - 1] > 0:
                alpha = 1.0 / d[j - 1]
            else:
                alpha = 0.0
            for k in range(1, j):
                beta = M[k - 1, j - 1]
                U[k - 1, j - 1] = alpha * beta
                M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
        d[0] = M[0, 0]
        return U, np.diag(d)

    @staticmethod
    def thornton(xin, Phi, Uin, Din, Gin, Q):
        x, U, D = Phi @ xin, Uin, Din
        n, r = 3, 3
        G = Gin
        U = np.eye(3)
        PhiU = Phi @ Uin
        for i in reversed(range(3)):
            sigma = 0
            for j in range(n):
                sigma = sigma + PhiU[i,j]**2 * Din[j,j]
                if (j <= r-1):
                    sigma = sigma + G[i,j]**2 + Q[j,j]
            D[i,i] = sigma
            ilim = i-1
            if not ilim < 0:
                for j in range(ilim):
                    sigma = 0
                    for k in range(n):
                        sigma = sigma + PhiU[i,k] * Din[k,k] * PhiU[j,k]
                    for k in range(r):
                        sigma = sigma + G[i,k] * Q[k,k] * G[j,k]
                    U[j,i] = sigma / D[i,i]
                    for k in range(n):
                        PhiU[j,k] = PhiU[j,k] - U[j,i] * PhiU[i,k]
                    for k in range(r):
                        G[j,k] = G[j,k] - U[j,i] * G[i,k]
        return x, U, D

    @staticmethod
    def bierman(z, R, H, xin, Uin, Din):
        x, U, D = xin, Uin, Din
        a = U.T @ H.T
        b = D @ a
        dz = z # z - H @ xin
        alpha = R
        gamma = 1 / alpha
        for j in range(3):
            beta = alpha
            alpha = alpha + a[j] * b[j]
            lamda = -a[j] * gamma
            gamma = 1 / alpha
            D[j, j] = beta * gamma * D[j, j]
            jlim = j-1
            if not jlim < 0:
                for i in range(jlim):
                    beta = U[i, j]
                    U[i, j] = beta + b[i] * lamda
                    b[i] = b[i] + b[j] * beta
        dzs = gamma * dz
        x = x + dzs * b
        return x, U, D

    @staticmethod
    def div0(a, b):
        try:
            return a / float(b)
        except:
            return np.nan

if __name__ == "__main__":
    cl = Classical()
    cl.xbp1()
    cl.xbp2()
