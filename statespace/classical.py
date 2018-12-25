import numpy as np
from innov import Innov

def ud_decomposition(M):
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

def thornton_temporal_update(xin, Phi, Uin, Din, Gin, Q):
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

def bierman_observational_update(z, R, H, xin, Uin, Din):
    x, U, D = xin, Uin, Din
    a = U.T @ H.T
    b = D @ a
    dz = z  # z - H @ xin
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
    return x, U, D

class Classical():
    def __init__(self, mode, plot=True):
        self.log = []
        from sim import Sim1
        s= Sim1()
        if mode == 'linearized':
            self.sim1_linearized(s)
        elif mode == 'extended':
            self.sim1_extended(s)
        if mode == 'adaptive':
            from sim import Sim1b
            s = Sim1b()
            self.sim1b_adaptive(s)
        innov = Innov(self.log)
        if plot: innov.plot_standard()

    def sim1_linearized(self, s):
        xref, xhat, Ptil = 2., 2.2, .01
        for step in s.steps():
            xref = 2. + .067 * step[0]
            xhat = s.a(xref, 0) + s.A(xref) * (xhat - xref)
            Ptil = s.A(xref) * Ptil * s.A(xref)
            Ree = s.C(xref) * Ptil * s.C(xref) + s.Rvv
            K = Ptil * s.C(xref) / Ree
            yhat = s.c(xref, 0) + s.C(xref) * (xhat - xref)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * s.C(xref)) * Ptil
            self.log.append([step[0], xhat, yhat, step[1]-xhat, step[2]-yhat])

    def sim1_extended(self, s):
        xref, xhat, Ptil = 2., 2.2, .01
        for step in s.steps():
            xhat = s.a(xhat, 0)
            Ptil = s.A(xhat) * Ptil * s.A(xhat)
            Ree = s.C(xhat) * Ptil * s.C(xhat) + s.Rvv
            K = Ptil * s.C(xhat) / Ree
            yhat = s.c(xhat, 0)
            xhat = xhat + K * (step[2] - yhat)
            Ptil = (1 - K * s.C(xhat)) * Ptil
            self.log.append([step[0], xhat, yhat, step[1]-xhat, step[2]-yhat])

    def sim1b_adaptive(self, s):
        xhat = [2.2, .055, .044]
        Ptil = 100. * np.eye(3)
        U, D = ud_decomposition(Ptil)
        for step in s.steps():
            xhat = s.a(xhat, 0)
            Phi = s.A(xhat)
            xhat, U, D = thornton_temporal_update(xin=xhat, Phi=Phi, Uin=U, Din=D, Gin=np.eye(3), Q=s.Rww)
            yhat = s.c(xhat, 0)
            xhat, U, D = bierman_observational_update(z=step[2] - yhat, R=s.Rvv, H=s.C(xhat), xin=xhat, Uin=U, Din=D)
            self.log.append([step[0], xhat[0], yhat, step[1][0]-xhat[0], step[2]-yhat])

if __name__ == "__main__":
    Classical('linearized')
    Classical('extended')
    Classical('adaptive')
