import math, util
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from scipy.linalg.blas import drot, drotg
import sys; sys.path.append('../')
from models.threestate import Threestate
from models.onestate import Onestate
from innovations import Innovs

def main():
    processor = Modern()
    # model = Onestate()
    model = Threestate()
    # processor.spkf(model)
    processor.spkfcholeksy(model)
    processor.innovs.plot()

class Modern():
    '''modern sigma-point or ukf filter'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.innovs = args, kwargs, Innovs()

    def spkf(self, model):
        '''sigma-point sampling kalman filter'''
        steps, f, h, F, H, R, x, P = model.ekf()
        f, h, X1, X2, W = model.sp()
        for t, xt, yt in steps():
            X = f(X1(x, P))
            Y = h(X2(X))
            x = W @ X
            y = W @ Y
            P = W @ np.power(X - x, 2) + model.Rww
            K = (W @ np.multiply(X - x, Y - y)) / (W @ np.power(Y - y, 2) + R)
            x = x + K * (yt - y)
            P = P - K * (W @ np.power(Y - y, 2) + R) * K
            self.innovs.add(t, xt, yt, x, y)

    def spkfcholeksy(self, model):
        '''cholesky factorized sigma-point sampling kalman filter'''
        steps, f, h, F, H, R, x, P = model.ekf()
        f, h, X1, X2, W, S = model.sp()
        for t, xt, yt in steps():
            x, S, X = self.temporal(x, S, model)
            x, S, y = self.observational(x, S, X, yt, model)
            self.innovs.add(t, xt, yt, x, y)

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

    def temporal(self, xhat, S, model):
        '''cholesky factorized temporal update'''
        X = model.SPKF.vf(model.SPKF.X1(xhat, S))
        xhat = model.SPKF.Wm @ X.T
        for i in range(7): model.SPKF.Xtil[:, i] = X[:, i] - xhat
        q, r = np.linalg.qr(np.concatenate([math.sqrt(model.SPKF.Wc[1]) * model.SPKF.Xtil[:, 1:], model.SPKF.Sw], 1))
        S = self.cholupdate(r.T[0:3, 0:3], model.SPKF.Wc[0] * model.SPKF.Xtil[:, 0])
        return xhat, S, X

    def observational(self, xhat, S, X, obs, model):
        '''cholesky factorized observational update'''
        Y = model.SPKF.vh(model.SPKF.X2(X))
        yhat = model.SPKF.Wm @ Y.T
        for i in range(7): model.SPKF.Ytil[0, i] = Y[i] - yhat
        q, r = np.linalg.qr(np.concatenate([math.sqrt(model.SPKF.Wc[1]) * model.SPKF.Ytil[:, 1:], model.SPKF.Sv], 1))
        Sy = self.cholupdate(r.T[0:1, 0:1], model.SPKF.Wc[0] * model.SPKF.Ytil[:, 0])
        for i in range(7): model.SPKF.Pxy[:, 0] = model.SPKF.Pxy[:, 0] + model.SPKF.Wc[i] * model.SPKF.Xtil[:, i] * model.SPKF.Ytil.T[i, :]
        if Sy[0, 0] < math.sqrt(10) or Sy[0, 0] > math.sqrt(1000): Sy[0, 0] = math.sqrt(1000)
        K = model.SPKF.Pxy / Sy[0, 0] ** 2
        U = K * Sy
        xhat = xhat + K[:, 0] * (obs - yhat)
        S = self.choldowndate(S, U)
        return xhat, S, yhat

if __name__ == "__main__":
    main()

