import numpy as np

class Resample():
    def __init__(self):
        pass

    def invcdf(self, xi, Wi):
        tmp = []
        for ndx in range(xi.size):
            tmp.append([xi[ndx], Wi[ndx]])
        tmp = sorted(tmp, key=lambda x: x[0])
        cdf = []
        xprv, Wprv = tmp[0][0], tmp[0][1]
        cdf.append([xprv, Wprv])
        cdfndx = 0
        for i in range(1, len(tmp)):
            if abs(tmp[i][0] - xprv) > 1e-5:
                cdf[cdfndx][1] = Wprv
                xprv, Wprv = tmp[i][0], tmp[i][1]
                cdf.append([xprv, Wprv])
                cdfndx += 1
            else:
                Wprv += tmp[i][1]
        cdf = np.asarray(cdf)
        cdfsum = np.sum(cdf, axis=0)
        cdfmen = np.mean(cdf, axis=0)
        uk = np.sort(np.random.uniform(size=xi.size))
        xhati = xi
        Whati = Wi
        return xhati, Whati
