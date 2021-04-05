import math
import numpy as np
from scipy.linalg.blas import drot, drotg

class SigmaPoint():
    '''modern sigma-point or ukf filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []
        if 'cho' in args: self.run = self.cho
        else: self.run = self.base

    def base(self, model):
        '''sigma-point sampling kalman filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        f, h, Xtil, Ytil, X1, X2, Pxy, W, Wc, S, Sproc, Sobs = model.sp()
        for t, o, u in sim():
            X = f(X1(x, P))
            Y = h(X2(X))
            x = W @ X.T
            y = W @ Y
            P = np.diag(W @ np.power(self.huh(X.copy(), x), 2).T + model.varproc)
            tmp = np.multiply(self.huh(X.copy(), x), Y - y)
            K = W @ tmp.T / (W @ np.power(Y - y, 2) + R)
            x = x + K * (o - y)
            P = P - K * (W @ np.power(Y - y, 2) + R) * K
            self.log.append([t, x, y])

    def huh(self, X, x):
        for i in range(7): X[:, i] -= x
        return X


    def cho(self, model):
        '''cholesky factorized sigma-point sampling kalman filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        f, h, Xtil, Ytil, X1, X2, Pxy, W, Wc, S, Sproc, Sobs = model.sp()
        for t, o, u in sim():
            x, S, X = self.temporal(x, f, Xtil, X1, W, Wc, S, Sproc)
            x, S, y = self.observational(x, o, h, X, Xtil, Ytil, X2, Pxy, W, Wc, S, Sobs)
            self.log.append([t, x, y])

    def cholupdate(self, R, z):
        '''cholesky update'''
        n = z.shape[0]
        for k in range(n):
            c, s = drotg(R[k, k], z[k])
            drot(R[k, :], z, c, s, overwrite_x=True, overwrite_y=True)
        return R

    def choldowndate(self, R, z):
        '''cholesky downdate'''
        n = R.shape[0]
        for k in range(n):
            if (R[k, k] - z[k]) * (R[k, k] + z[k]) < 0: return R
            rbar = np.sqrt((R[k, k] - z[k]) * (R[k, k] + z[k]))
            for j in range(k + 1, n):
                R[k, j] = 1. / rbar * (R[k, k] * R[k, j] - z[k] * z[j])
                z[j] = 1. / R[k, k] * (rbar * z[j] - z[k] * R[k, j])
            R[k, k] = rbar
        return R

    def temporal(self, x, f, Xtil, X1, Wm, Wc, S, Sproc):
        '''cholesky factorized temporal update'''
        X = f(X1(x, S))
        x = Wm @ X.T
        for i in range(7): Xtil[:, i] = X[:, i] - x
        q, r = np.linalg.qr(np.concatenate([math.sqrt(Wc[1]) * Xtil[:, 1:], Sproc], 1))
        S = self.cholupdate(r.T[0:3, 0:3], Wc[0] * Xtil[:, 0])
        return x, S, X

    def observational(self, x, o, h, X, Xtil, Ytil, X2, Pxy, Wm, Wc, S, Sobs):
        '''cholesky factorized observational update'''
        Y = h(X2(X))
        y = Wm @ Y.T
        for i in range(7): Ytil[0, i] = Y[i] - y
        q, r = np.linalg.qr(np.concatenate([math.sqrt(Wc[1]) * Ytil[:, 1:], Sobs], 1))
        Sy = self.cholupdate(r.T[0:1, 0:1], Wc[0] * Ytil[:, 0])
        for i in range(7): Pxy[:, 0] = Pxy[:, 0] + Wc[i] * Xtil[:, i] * Ytil.T[i, :]
        if Sy[0, 0] < math.sqrt(10) or Sy[0, 0] > math.sqrt(1000): Sy[0, 0] = math.sqrt(1000)
        K = Pxy / Sy[0, 0] ** 2
        U = K * Sy
        x = x + K[:, 0] * (o - y)
        S = self.choldowndate(S, U)
        return x, S, y

if __name__ == "__main__":
    pass

