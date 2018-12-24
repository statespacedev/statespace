import numpy as np
import math
from innov import Innov

class Classical():
    def __init__(self):
        self.log = []

    def sim1_linearized(self):
        from sim import Sim1
        sim = Sim1()
        dt = sim.dt
        def c(x): return x**2 + x**3
        def A(x): return 1 - .05*dt + .08*dt*x
        def C(x): return 2*x + 3*x**2
        xref, xhat, Ptil = 2., 2.2, .01
        for step in sim.steps():
            if step[0] == 0:
                self.log.append([step[0], xhat, c(xhat), step[1]-xhat, step[2]-c(xhat)])
                continue
            xhat = ((1-.05*dt)*xref + .04*dt*xref**2) + A(xref) * (xhat - xref)
            Ptil = A(xref)**2 * Ptil
            xref = 2. + .067 * step[0]
            Ree = C(xhat)**2*Ptil + sim.Rvv
            K = Ptil*C(xref)/Ree
            yhat = (xref**2 + xref**3) + C(xref) * (xhat - xref)
            e = step[2]-yhat
            xhat = xhat + K*e
            Ptil = (1 - K*C(xref)) * Ptil
            xtil = step[1]-xhat
            self.log.append([step[0], xhat, yhat, xtil, e])
        innov = Innov(self.log)
        innov.plot_standard()

    def sim1_extended(self):
        from sim import Sim1
        sim = Sim1()
        dt = sim.dt
        def a(x): return (1 - .05*dt)*x + (.04*dt)*x**2
        def c(x): return x**2 + x**3
        def A(x): return 1 - .05*dt + .08*dt*x
        def C(x): return 2*x + 3*x**2
        xref, xhat, Ptil = 2., 2.2, .01
        for step in sim.steps():
            if step[0] == 0:
                self.log.append([step[0], xhat, c(xhat), step[1]-xhat, step[2]-c(xhat)])
                continue
            Phi = A(a(xhat))
            xhat = a(xhat)
            Ptil = Phi*Ptil*Phi
            yhat = c(xhat)
            e = step[2]-yhat
            Ree = C(xhat)*Ptil*C(xhat) + sim.Rvv
            K = Ptil*C(xhat)/Ree
            xhat = xhat + K*e
            Ptil = (1 - K*C(xhat))*Ptil
            xtil = step[1]-xhat
            self.log.append([step[0], xhat, yhat, xtil, e])
        innov = Innov(self.log)
        innov.plot_standard()

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

if __name__ == "__main__":
    cl = Classical()
    cl.sim1_linearized()
    cl = Classical()
    cl.sim1_extended()
    # cl.xbp2()
