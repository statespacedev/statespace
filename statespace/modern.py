import math, util
import numpy as np
from scipy.linalg.blas import drot, drotg
import sys; sys.path.append('../')
from models.threestate import Threestate
from models.onestate import Onestate

def main():
    processor = Modern()
    # model = Onestate()
    model = Threestate()
    # processor.spkf1(model)
    processor.spkf2(model)
    processor.innov.plot()

class Modern():
    '''sigma-point or ukf filter. the run methods bring in particular models from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods.'''
    
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.innov = util.Innovs()

    def spkf1(self, model):
        '''sigma-point kalman filter 1.'''
        xhat, Ptil = model.x0, model.P0
        for step in model.steps():
            X = model.spkf.va(model.spkf.X(xhat, Ptil))
            Y = model.spkf.vc(model.spkf.Xhat(X))
            Ptil = model.spkf.W @ np.power(X - model.spkf.W @ X, 2) + model.Rww
            Rksiksi = model.spkf.W @ np.power(Y - model.spkf.W @ Y, 2) + model.R
            RXtilksi = model.spkf.W @ np.multiply(X - model.spkf.W @ X, Y - model.spkf.W @ Y)
            K = RXtilksi / Rksiksi
            yhat = model.spkf.W @ Y
            xhat = model.spkf.W @ X + K * (step[2] - model.spkf.W @ Y)
            Ptil = Ptil - K * Rksiksi * K
            self.innov.add(step[0], xhat, yhat, step[1] - xhat, step[2] - yhat)

    def spkf2(self, model):
        '''sigma-point kalman filter 2.'''
        xhat = np.array([2.0, .055, .044])
        S = np.linalg.cholesky(.1 * np.eye(3))
        for step in model.steps():
            xhat, S, X = self.temporal_update(xhat=xhat, S=S, model=model)
            xhat, S, yhat = self.observational_update(xhat=xhat, S=S, X=X, obs=step[2], model=model)
            self.innov.add(step[0], xhat[0], yhat, step[1][0] - xhat[0], step[2] - yhat)

    def cholupdate(self, R, z):
        '''cholesky update.'''
        n = z.shape[0]
        for k in range(n):
            c, s = drotg(R[k, k], z[k])
            drot(R[k, :], z, c, s, overwrite_x=True, overwrite_y=True)
        return R

    def choldowndate(self, R, z):
        '''cholesky downdate.'''
        n = R.shape[0]
        for k in range(n):
            if (R[k, k] - z[k]) * (R[k, k] + z[k]) < 0: return R
            rbar = np.sqrt((R[k, k] - z[k]) * (R[k, k] + z[k]))
            for j in range(k + 1, n):
                R[k, j] = 1. / rbar * (R[k, k] * R[k, j] - z[k] * z[j])
                z[j] = 1. / R[k, k] * (rbar * z[j] - z[k] * R[k, j])
            R[k, k] = rbar
        return R

    def temporal_update(self, xhat, S, model):
        '''temporal update.'''
        X = model.spkf.va(model.spkf.X(xhat, S))
        xhat = model.spkf.Wm @ X.T
        for i in range(7): model.spkf.Xtil[:, i] = X[:, i] - xhat
        q, r = np.linalg.qr(np.concatenate([math.sqrt(model.spkf.Wc[1]) * model.spkf.Xtil[:, 1:], model.spkf.Sw], 1))
        S = self.cholupdate(r.T[0:3, 0:3], model.spkf.Wc[0] * model.spkf.Xtil[:, 0])
        return xhat, S, X

    def observational_update(self, xhat, S, X, obs, model):
        '''observational update.'''
        Y = model.spkf.vc(model.spkf.Xhat(X))
        yhat = model.spkf.Wm @ Y.T
        for i in range(7): model.spkf.Ytil[0, i] = Y[i] - yhat
        q, r = np.linalg.qr(np.concatenate([math.sqrt(model.spkf.Wc[1]) * model.spkf.Ytil[:, 1:], model.spkf.Sv], 1))
        Sy = self.cholupdate(r.T[0:1, 0:1], model.spkf.Wc[0] * model.spkf.Ytil[:, 0])
        for i in range(7): model.spkf.Pxy[:, 0] = model.spkf.Pxy[:, 0] + model.spkf.Wc[i] * model.spkf.Xtil[:, i] * model.spkf.Ytil.T[i, :]
        if Sy[0, 0] < math.sqrt(10) or Sy[0, 0] > math.sqrt(1000): Sy[0, 0] = math.sqrt(1000)
        K = model.spkf.Pxy / Sy[0, 0] ** 2
        U = K * Sy
        xhat = xhat + K[:, 0] * (obs - yhat)
        S = self.choldowndate(S, U)
        return xhat, S, yhat

if __name__ == "__main__":
    main()

