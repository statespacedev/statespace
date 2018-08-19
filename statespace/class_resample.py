import numpy as np

class Resample():
    def __init__(self, tol):
        self.tol = tol

    def invcdf(self, xi, Wi):
        tmp = []
        for ndx in range(xi.size):
            tmp.append([xi[ndx], Wi[ndx]])
        tmp = sorted(tmp, key=lambda x: x[0])
        cdf = [[tmp[0][0], tmp[0][1]]]
        cdfndx = 0
        for i in range(1, len(tmp)):
            if abs(tmp[i][0] - tmp[i-1][0]) > self.tol:
                cdf.append([tmp[i][0], tmp[i][1] + cdf[cdfndx][1]])
                cdfndx += 1
            else:
                cdf[cdfndx][1] += tmp[i][1]
        cdf = np.asarray(cdf)
        cdfsum = np.sum(cdf, axis=0)
        cdfmen = np.mean(cdf, axis=0)
        uk = np.sort(np.random.uniform(size=xi.size))
        xhati, Whati, k = [], [], 0
        for row in cdf:
            while k < uk.size and uk[k] <= row[1]:
                xhati.append(row[0])
                Whati.append(1/float(xi.size))
                k += 1
        xhati = np.asarray(xhati)
        Whati = np.asarray(Whati)

        assert xhati.size == xi.size
        return xhati, Whati
