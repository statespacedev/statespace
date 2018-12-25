import numpy as np
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
            xhat = ((1-.05*dt)*xref + .04*dt*xref**2) + A(xref) * (xhat - xref)
            Ptil = A(xref)**2 * Ptil
            xref = 2. + .067 * step[0]
            Ree = C(xhat)**2*Ptil + sim.Rvv
            K = Ptil*C(xref)/Ree
            yhat = (xref**2 + xref**3) + C(xref) * (xhat - xref)
            xhat = xhat + K*(step[2]-yhat)
            Ptil = (1 - K*C(xref)) * Ptil
            self.log.append([step[0], xhat, yhat, step[1]-xhat, step[2]-yhat])
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
            Phi = A(a(xhat))
            xhat = a(xhat)
            Ptil = Phi*Ptil*Phi
            yhat = c(xhat)
            Ree = C(xhat)*Ptil*C(xhat) + sim.Rvv
            K = Ptil*C(xhat)/Ree
            xhat = xhat + K*(step[2]-yhat)
            Ptil = (1 - K*C(xhat))*Ptil
            self.log.append([step[0], xhat, yhat, step[1]-xhat, step[2]-yhat])
        innov = Innov(self.log)
        innov.plot_standard()

    def sim1b_adaptive(self):
        from sim import Sim1b
        sim = Sim1b()
        dt = sim.dt
        def a(x): return np.array([(1 - x[1]*dt)*x[0] + x[2]*dt*x[0]**2, x[1], x[2]])
        def c(x): return x[0]**2 + x[0]**3
        def A(x):
            A = np.eye(3)
            A[0, 0] = 1 - x[1]*dt + 2*x[2]*dt*x[0]
            A[0, 1] = -dt*x[0]
            A[0, 2] = dt*x[0]**2
            return A
        def C(x): return np.array([2*x[0] + 3*x[0]**2, 0, 0])
        xhat = [2.2, .055, .044]
        Ptil = 100. * np.eye(3)
        U, D = Classical.UD(Ptil)
        for step in sim.steps():
            Phi = A(a(xhat))
            yhat = c(xhat)
            xhat, U, D = Classical.thornton(xin=xhat, Phi=Phi, Uin=U, Din=D, Gin=np.eye(3), Q=sim.Rww)
            xhat, U, D = Classical.bierman(z=step[2]-yhat, R=sim.Rvv, H=C(xhat), xin=xhat, Uin=U, Din=D)
            self.log.append([step[0], xhat[0], yhat, step[1][0]-xhat[0], step[2]-yhat])
        innov = Innov(self.log)
        innov.plot_standard()

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
    cl = Classical()
    cl.sim1b_adaptive()
