import numpy as np
import statespace.util as util
import sys; sys.path.append('../'); sys.path.append('../cmake-build-debug/libstatespace')
from models.threestate import Threestate
from models.onestate import Onestate
import libstatespace
api = libstatespace.Api()

def main():
    processor = Classical()
    model = Onestate()
    # model = Threestate()
    processor.ekf(model)
    # processor.ekfud(model)
    try: processor.innovs.plot()
    except: pass

class Classical():
    '''classical kalman filter'''

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.innovs = util.Innovs()

    def ekf(self, model):
        '''basic form'''
        steps, a, c, A, C, Ro, xc, P = model.pieces()
        for t, x, y in steps():
            xc = a(xc)
            yc = c(xc)
            P = A(xc) @ P @ A(xc)
            K = P @ C(xc) / (C(xc) @ P @ C(xc) + Ro)
            xc = xc + K * (y - yc)
            P = (1 - K * C(xc)) * P
            self.innovs.add(t, x, y, xc, yc)

    def ekfud(self, m):
        '''UD factorized form'''
        xhat, Ptil = m.xhat0, m.Ptil0
        U, D = self.udfactorize(Ptil)
        for step in m.steps():
            xhat, U, D = self.temporal(xin=m.a(xhat), Uin=U, Din=D, Phi=m.A(xhat), Gin=m.G, Q=m.Q)
            xhat, U, D, yhat = self.observational(xin=xhat, Uin=U, Din=D, H=m.C(xhat), obs=step[2], R=m.Rvv, yhat=m.c(xhat))
            self.innovs.add(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def ekfudcpp(self, m):
        '''UD factorized form in cpp'''
        xhat, Ptil = m.xhat0, m.Ptil0
        ud = api.udfactorize(Ptil); U, D = ud[0], np.diag(ud[1].transpose()[0])
        for step in m.steps():
            res = api.temporal(xin=m.a(xhat), Uin=U, Din=D, Phi=m.A(xhat), Gin=m.G, Q=m.Q)
            xhat, U, D = res[0].flatten(), res[1], res[2]
            res = api.observational(xin=xhat, Uin=U, Din=D, H=m.C(xhat), obs=step[2], R=m.Rvv, yhat=m.c(xhat))
            xhat, U, D, yhat = res[0].flatten(), res[1], res[2], m.c(xhat)
            self.innovs.add(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def udfactorize(self, M):
        '''UD factorization'''
        assert np.allclose(M, M.T)
        n, M = M.shape[0], np.triu(M)
        U, d = np.eye(n), np.zeros(n)
        for j in reversed(range(2, n + 1)):
            d[j - 1] = M[j - 1, j - 1]
            if d[j - 1] > 0: alpha = 1.0 / d[j - 1]
            else: alpha = 0.0
            for k in range(1, j):
                beta = M[k - 1, j - 1]
                U[k - 1, j - 1] = alpha * beta
                M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
        d[0] = M[0, 0]
        return U, np.diag(d)

    def temporal(self, xin, Uin, Din, Phi, Gin, Q):
        '''thornton temporal update'''
        U, D, G, n, r = np.eye(len(xin)), Din, Gin, len(xin), len(xin)
        x, PhiU = Phi @ xin, Phi @ Uin
        for i in reversed(range(len(xin))):
            sigma = 0
            for j in range(n):
                sigma = sigma + PhiU[i, j] ** 2 * Din[j, j]
                if (j <= r - 1): sigma = sigma + G[i, j] ** 2 + Q[j, j]
            D[i, i] = sigma
            ilim = i - 1
            if ilim > 0:
                for j in range(ilim):
                    sigma = 0
                    for k in range(n): sigma = sigma + PhiU[i, k] * Din[k, k] * PhiU[j, k]
                    for k in range(r): sigma = sigma + G[i, k] * Q[k, k] * G[j, k]
                    U[j, i] = sigma / D[i, i]
                    for k in range(n): PhiU[j, k] = PhiU[j, k] - U[j, i] * PhiU[i, k]
                    for k in range(r): G[j, k] = G[j, k] - U[j, i] * G[i, k]
        return x, U, D

    def observational(self, xin, Uin, Din, H, obs, R, yhat):
        '''bierman observation update'''
        x, U, D, dz, alpha, gamma = xin, Uin, Din, obs - yhat, R, 1/R
        a = U.T @ H.T
        b = D @ a
        for j in range(len(xin)):
            beta = alpha
            alpha = alpha + a[j] * b[j]
            lamda = -a[j] * gamma
            gamma = 1 / alpha
            D[j, j] = beta * gamma * D[j, j]
            jlim = j - 1
            if not jlim < 0:
                for i in range(jlim):
                    beta = U[i, j]
                    U[i, j] = beta + b[i] * lamda
                    b[i] = b[i] + b[j] * beta
        dzs = gamma * dz
        x = x + dzs * b
        return x, U, D, yhat

if __name__ == "__main__":
    main()

