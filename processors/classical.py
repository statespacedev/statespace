import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import sys; sys.path.append('../'); sys.path.append('../cmake-build-debug/libstatespace')
from models.threestate import Threestate
from models.onestate import Onestate
from models.bearingsonly import BearingsOnly

def main():
    processor = Classical()
    model = BearingsOnly()
    # model = Onestate()
    # model = Threestate()
    processor.ekf(model)
    # processor.ekfud(model)
    model.eval.plot_model()

class Classical():
    '''classical kalman filter'''

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []

    def ekf(self, model):
        '''basic form'''
        sim, f, h, F, H, R, xe, P = model.ekf()
        for t, x, y in sim():
            xe = f(xe)
            ye = h(xe)
            P = F(xe) @ P @ F(xe)
            K = P @ H(xe) / (H(xe) @ P @ H(xe) + R)
            xe = xe + K * (y - ye)
            P = (1 - K * H(xe)) * P
            self.log.append([t, xe, ye])

    def ekfud(self, model):
        '''UD factorized form'''
        sim, f, h, F, H, R, xe, P = model.ekf()
        G, Q = model.ekfud()
        U, D = self.udfactorize(P)
        for t, x, y in sim():
            xe, U, D = self.temporal(f(xe), U, D, F(xe), G, Q)
            xe, U, D, ye = self.observational(xe, U, D, H(xe), y, R, h(xe))
            self.log.append([t, xe, ye])

    def ekfudcpp(self, model):
        '''UD factorized form in cpp'''
        import libstatespace
        api = libstatespace.Api()
        sim, f, h, F, H, R, xe, P = model.ekf()
        G, Q = model.ekfud()
        ud = api.udfactorize(P); U, D = ud[0], np.diag(ud[1].transpose()[0])
        for t, x, y in sim():
            cpp = api.temporal(f(xe), U, D, H(xe), G, Q)
            xe, U, D = cpp[0].flatten(), cpp[1], cpp[2]
            cpp = api.observational(xe, U, D, H(xe), y, R, h(xe))
            xe, U, D, ye = cpp[0].flatten(), cpp[1], cpp[2], h(xe)
            self.log.append([t, xe, ye])

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
