# extended bayesian processor, extended kalman filter, XBP, EKF
import numpy as np
import util, math, plots
n = 150
deltat = .01

def x(x, w):
    return (1 - .05 * deltat) * x + .04 * deltat * x**2 + w

def y(x, v):
    return x**2 + x**3 + v

def A(x):
    return 1 - .05 * deltat + .08 * deltat * x

def C(x):
    return 2 * x + 3 * x**2

def main():
    tts = np.arange(0, n * deltat, deltat)
    Rww = 0
    wts = math.sqrt(Rww) * np.random.randn(n)
    Rvv = .09
    vts = math.sqrt(Rvv) * np.random.randn(n)

    xts = np.zeros_like(tts)
    xts[0] = 2.
    yts = np.zeros_like(tts)
    yts[0] = y(xts[0], vts[0])

    xhatts = np.zeros_like(tts)
    xhatts[0] = 2.
    Ptilts = np.zeros_like(tts)
    Ptilts[0] = .01

    Reets = np.zeros_like(tts)
    Reets[0] = 0
    Kts = np.zeros_like(tts)
    Kts[0] = 0

    yhatts = np.zeros_like(tts)
    yhatts[0] = y(xhatts[0], 0)
    ets = np.zeros_like(tts)
    ets[0] = 0

    xtilts = np.zeros_like(tts)
    xtilts[0] = xhatts[0] - xts[0]

    for tk in range(1, n):
        xts[tk] = x(xts[tk - 1], wts[tk - 1])
        yts[tk] = y(xts[tk], vts[tk])

        xhatts[tk] = x(xhatts[tk-1], 0)
        Ptilts[tk] = A(xhatts[tk-1])**2 * Ptilts[tk-1]

        Reets[tk] = C(xhatts[tk])**2 * Ptilts[tk] + .09
        Kts[tk] = util.div0( Ptilts[tk] * C(xhatts[tk]) , Reets[tk] )

        yhatts[tk] = y(xhatts[tk], 0)
        ets[tk] = yts[tk] - yhatts[tk]

        xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
        Ptilts[tk] = (1 - Kts[tk] * C(xhatts[tk])) * Ptilts[tk]

        xtilts[tk] = xhatts[tk] - xts[tk]

    plots.test(tts, xhatts, xtilts, yhatts, ets, yts, Reets)

if __name__ == "__main__":
    main()

