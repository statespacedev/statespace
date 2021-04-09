import math
import numpy as np
from scipy.linalg.blas import drot, drotg

class SigmaPoint():
    '''modern sigma-point deterministic sampling kalman filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.log = args, kwargs, []
        if 'cho' in args: self.run = self.spfcholesky
        else: self.run = self.spfbase

    def spfbase(self, model):
        '''sigma-point determinstic sampling kalman filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        f, h, Xtil, Ytil, X1, X2, Pxy, W = model.sp()
        Wm = np.tile(W, (Xtil.shape[0], 1))
        for t, o, u in sim():
            X = f(X1(x, P))
            Y = h(X2(X))
            x, y = np.sum(np.multiply(Wm, X), axis=1).reshape(-1, 1), np.sum(np.multiply(W, Y), axis=1)[0]
            Xres, Yres = self.Xres(X.copy(), x), self.Yres(Y, y)
            P = np.multiply(Wm, Xres) @ Xres.T + G @ Q @ G.T
            K = np.multiply(Wm, Xres) @ Yres.T / (np.multiply(W, Yres) @ Yres.T + R)
            x = x + K * (o - y)
            P = P - K @ (np.multiply(W, Yres) @ Yres.T + R) @ K.T
            self.log.append([t, x, y])

    def spfcholesky(self, model):
        '''cholesky factorized sigma-point sampling kalman filter'''
        sim, f, h, F, H, R, Q, G, x, P = model.ekf()
        f, h, Xtil, Ytil, X1, X2, Pxy, W, Wc, S, Sproc, Sobs = model.spcho()
        for t, o, u in sim():
            x, S, X = self.temporal(x, f, Xtil, X1, W, Wc, S, Sproc)
            x, S, y = self.observational(x, o, h, X, Xtil, Ytil, X2, Pxy, W, Wc, S, Sobs)
            self.log.append([t, x, y])

    def Xres(self, X, x):
        for i in range(X.shape[1]):
            for j in range(X.shape[0]): X[j, i] -= x[j, 0]
        return X

    def Yres(self, Y, y):
        for i in range(Y.shape[1]): Y[0, i] -= y
        return Y

    def temporal(self, x, f, Xtil, X1, Wm, Wc, S, Sproc):
        '''cholesky factorized temporal update'''
        X = f(X1(x, S))
        x = X @ Wm.T
        for i in range(X.shape[1]): Xtil[:, i] = (X[:, i].reshape(-1, 1) - x).T
        q, r = np.linalg.qr(np.concatenate([math.sqrt(Wc[0, 1]) * Xtil[:, 1:], Sproc], 1))
        S = self.cholupdate(r.T[0:X.shape[0], 0:X.shape[0]], Wc[0, 0] * Xtil[:, 0].reshape(-1, 1))
        return x, S, X

    def observational(self, x, o, h, X, Xtil, Ytil, X2, Pxy, Wm, Wc, S, Sobs):
        '''cholesky factorized observational update'''
        Y = h(X2(X))
        y = (Wm @ Y.T)[0, 0]
        for i in range(X.shape[1]): Ytil[0, i] = (Y[0, i].reshape(-1, 1) - y).T
        q, r = np.linalg.qr(np.concatenate([math.sqrt(Wc[0, 1]) * Ytil[:, 1:], Sobs], 1))
        Sy = self.cholupdate(r.T[0:1, 0:1], Wc[0] * Ytil[:, 0].reshape(-1, 1))
        for i in range(X.shape[1]):
            tmp = Wc[0, i] * Xtil[:, i].reshape(-1, 1) * Ytil[0, i]
            Pxy += tmp
        if Sy[0, 0] < math.sqrt(10) or Sy[0, 0] > math.sqrt(1000): Sy[0, 0] = math.sqrt(1000)
        K = Pxy / Sy[0, 0] ** 2
        U = K * Sy
        x = x + K * (o - y)
        S = self.choldowndate(S, U)
        return x, S, y

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

if __name__ == "__main__":
    pass
