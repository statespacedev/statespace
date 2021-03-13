import numpy as np
import statespace.util as util
import sys; sys.path.append('../'); sys.path.append('../cmake-build-debug/libstatespace')
from models.jazwinski2 import Jazwinski2
from models.jazwinski1 import Jazwinski1
import libstatespace
api = libstatespace.Api()

def main():
    # run('kf')
    # run('ekf')
    run('ekfud')

def run(mode='ekf'):
    '''individual 'run functions' here use particular versions of the processor, for example standard ekf and ud factorized ekf, and run the processor on a particular model problem, for example jazwinski1 or jazwinski2.'''
    processor = Classical()
    if mode == 'kf': processor.kf(Jazwinski1())
    elif mode == 'ekf': processor.ekf(Jazwinski1())
    elif mode == 'ekfud': processor.ekfud(Jazwinski2())
    processor.innov.plot()

class Classical():
    '''classical kalman filter.'''

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.innov = util.Innovs()

    def kf(self, model):
        '''basic kalman filter. linearized about a reference trajectory which has to be expressed explicitly. this is one of the distinguishing characteristics - in a basic kalman filter there has to be an explicit reference trajectory, in the extended kalman filter there's not.'''
        xhat, Ptil = model.xhat0, model.Ptil0
        for step in model.steps():
            xref = model.xref(step[0])
            xhat = model.a(xref, 0) + model.A(xref) * (xhat - xref)
            Ptil = model.A(xref) * Ptil * model.A(xref)
            Ree = model.C(xref) * Ptil * model.C(xref) + model.Rvv
            K = Ptil * model.C(xref) / Ree
            yhat = model.c(xref, 0) + model.C(xref) * (xhat - xref)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * model.C(xref)) * Ptil
            self.innov.update(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat)

    def ekf(self, model):
        '''standard form extended kalman filter.'''
        xhat, Ptil = model.xhat0, model.Ptil0
        for step in model.steps():
            xhat = model.a(xhat)
            Ptil = model.A(xhat) * Ptil * model.A(xhat)
            Ree = model.C(xhat) * Ptil * model.C(xhat) + model.Rvv
            K = Ptil * model.C(xhat) / Ree
            yhat = model.c(xhat)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * model.C(xhat)) * Ptil
            self.innov.update(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat)

    def ekfud(self, model):
        '''UD factorized form of the extended kalman filter, or square-root filter, with better numerical characteristics. instead of a covariance matrix full of squared values, we propagate something like it's square-root. this is the U matrix. this makes the state and observation equations look different, but they're doing the same thing as the standard form.'''
        xhat, Ptil = model.xhat0b, model.Ptil0b
        U, D = self.udfactorize(Ptil)
        for step in model.steps():
            xhat, U, D = self.temporal(xin=model.a(xhat), Uin=U, Din=D, Phi=model.A(xhat), Gin=model.G, Q=model.Q)
            xhat, U, D, yhat = self.observational(xin=xhat, Uin=U, Din=D, H=model.C(xhat), obs=step[2], R=model.Rvv, yhat=model.c(xhat))
            self.innov.update(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def ekfudcpp(self, model):
        '''UD factorized form of the extended kalman filter, in cpp.'''
        xhat, Ptil = model.xhat0b, model.Ptil0b
        ud = api.udfactorize(Ptil); U, D = ud[0], np.diag(ud[1].transpose()[0])
        for step in model.steps():
            res = api.temporal(xin=model.a(xhat), Uin=U, Din=D, Phi=model.A(xhat), Gin=model.G, Q=model.Q)
            xhat, U, D = res[0].flatten(), res[1], res[2]
            res = api.observational(xin=xhat, Uin=U, Din=D, H=model.C(xhat), obs=step[2], R=model.Rvv, yhat=model.c(xhat))
            xhat, U, D, yhat = res[0].flatten(), res[1], res[2], model.c(xhat)
            self.innov.update(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def udfactorize(self, M):
        '''ud factorization.'''
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
        '''thornton temporal update.'''
        U, D, G, n, r = np.eye(3), Din, Gin, 3, 3
        x, PhiU = Phi @ xin, Phi @ Uin
        for i in reversed(range(3)):
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
        '''bierman observation update.'''
        x, U, D, dz, alpha, gamma = xin, Uin, Din, obs - yhat, R, 1/R
        a = U.T @ H.T
        b = D @ a
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

