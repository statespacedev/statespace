# linearized bayesian processor, linearized kalman filter, LZ-BP
import numpy as np
import util, math
import class_innov

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
    xrefts = np.zeros_like(tts)
    xrefts[0] = 2.

    xhatts = np.zeros_like(tts)
    xhatts[0] = 2.2
    Ptilts = np.zeros_like(tts)
    Ptilts[0] = .01
    xtilts = np.zeros_like(tts)
    xtilts[0] = xts[0] - xhatts[0]

    Reets = np.zeros_like(tts)
    Reets[0] = 0
    Kts = np.zeros_like(tts)
    Kts[0] = 0

    yhatts = np.zeros_like(tts)
    yhatts[0] = y(xrefts[0], 0) + C(xrefts[0]) * (xhatts[0] - xrefts[0])
    ets = np.zeros_like(tts)
    ets[0] = yts[0] - yhatts[0]

    for tk in range(1, n):
        xts[tk] = x(xts[tk - 1], wts[tk - 1])
        yts[tk] = y(xts[tk], vts[tk])
        xrefts[tk] = 2. + .067 * tts[tk]

        xhatts[tk] = x(xrefts[tk-1], 0) + A(xrefts[tk-1]) * (xhatts[tk-1] - xrefts[tk-1])
        Ptilts[tk] = A(xrefts[tk-1])**2 * Ptilts[tk-1]

        Reets[tk] = C(xhatts[tk])**2 * Ptilts[tk] + Rvv
        Kts[tk] = util.div0( Ptilts[tk] * C(xrefts[tk]) , Reets[tk] )

        yhatts[tk] = y(xrefts[tk], 0) + C(xrefts[tk]) * (xhatts[tk] - xrefts[tk])
        ets[tk] = yts[tk] - yhatts[tk]

        xhatts[tk] = xhatts[tk] + Kts[tk] * ets[tk]
        Ptilts[tk] = (1 - Kts[tk] * C(xrefts[tk])) * Ptilts[tk]
        xtilts[tk] = xts[tk] - xhatts[tk]

    innov = class_innov.Innov(tts, ets)
    innov.standard(tts, xhatts, xtilts, yhatts)

if __name__ == "__main__":
    main()
