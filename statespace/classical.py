import numpy as np
import util
import sys; sys.path.append('../')
from models.jazwinski2 import Jazwinski2
from models.jazwinski1 import Jazwinski1
from models.rccircuit import Rccircuit

def main(mode='ekf2'):
    processor = Classical()
    if mode == 'kf1': processor.kf1(Rccircuit(signal=300))
    elif mode == 'kf2': processor.kf2(Jazwinski1())
    elif mode == 'ekf1': processor.ekf1(Jazwinski1())
    elif mode == 'ekf2': processor.ekf2(Jazwinski2())
    processor.innov.plot()

class Classical():
    '''classical kalman filter. the run methods bring in particular models from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods.'''

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.innov = util.Innovs()

    def kf1(self, model):
        '''rccircuit.'''
        xhat = 2.5
        Ptil = 50e-4
        for step in model.steps():
            xhat = .97 * xhat + 100 * model.u
            Ptil = .94 * Ptil + model.Rww
            Ree = 4 * Ptil + 4
            K = 2 * Ptil / Ree
            yhat = 2 * xhat
            xhat = xhat + K * (step[2] - yhat)
            Ptil = Ptil / (Ptil + 1)
            self.innov.update2(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat, Ree, Ptil)

    def kf2(self, model):
        '''kalman filter.'''
        xhat = 2.2
        Ptil = .01
        for step in model.steps():
            xref = 2. + .067 * step[0]
            xhat = model.a(xref, 0) + model.A(xref) * (xhat - xref)
            Ptil = model.A(xref) * Ptil * model.A(xref)
            Ree = model.C(xref) * Ptil * model.C(xref) + model.Rvv
            K = Ptil * model.C(xref) / Ree
            yhat = model.c(xref, 0) + model.C(xref) * (xhat - xref)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * model.C(xref)) * Ptil
            self.innov.update(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat)

    def ekf1(self, model):
        '''extended kalman filter 1.'''
        xhat = 2.2
        Ptil = .01
        for step in model.steps():
            xhat = model.a(xhat)
            Ptil = model.A(xhat) * Ptil * model.A(xhat)
            Ree = model.C(xhat) * Ptil * model.C(xhat) + model.Rvv
            K = Ptil * model.C(xhat) / Ree
            yhat = model.c(xhat)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * model.C(xhat)) * Ptil
            self.innov.update(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat)

    def ekf2(self, model):
        '''extended kalman filter 2.'''
        xhat = np.array([2, .055, .044])
        U, D = self.ud_factorization(1. * np.eye(3))
        for step in model.steps():
            xhat, U, D = self.temporal_update(xin=model.a(xhat), Uin=U, Din=D, model=model)
            xhat, U, D, yhat = self.observational_update(xin=xhat, Uin=U, Din=D, obs=step[2], model=model)
            self.innov.update(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def ud_factorization(self, M):
        '''ud factorization.'''
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


    def temporal_update(self, xin, Uin, Din, model):
        '''thornton temporal update.'''
        Phi, Gin, Q = model.A(xin), model.G, model.Q
        x, U, D = Phi @ xin, Uin, Din
        n, r = 3, 3
        G = Gin
        U = np.eye(3)
        PhiU = Phi @ Uin
        for i in reversed(range(3)):
            sigma = 0
            for j in range(n):
                sigma = sigma + PhiU[i, j] ** 2 * Din[j, j]
                if (j <= r - 1):
                    sigma = sigma + G[i, j] ** 2 + Q[j, j]
            D[i, i] = sigma
            ilim = i - 1
            if not ilim < 0:
                for j in range(ilim):
                    sigma = 0
                    for k in range(n):
                        sigma = sigma + PhiU[i, k] * Din[k, k] * PhiU[j, k]
                    for k in range(r):
                        sigma = sigma + G[i, k] * Q[k, k] * G[j, k]
                    U[j, i] = sigma / D[i, i]
                    for k in range(n):
                        PhiU[j, k] = PhiU[j, k] - U[j, i] * PhiU[i, k]
                    for k in range(r):
                        G[j, k] = G[j, k] - U[j, i] * G[i, k]
        return x, U, D


    def observational_update(self, xin, Uin, Din, obs, model):
        '''bierman observation update.'''
        R, H, yhat = model.Rvv, model.C(xin), model.c(xin)
        x, U, D = xin, Uin, Din
        a = U.T @ H.T
        b = D @ a
        dz = obs - yhat  # z - H @ xin
        alpha = R
        gamma = 1 / alpha
        for j in range(3):
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

