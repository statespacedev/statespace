import numpy as np
import matplotlib.pyplot as plt
from pymc3.plots.kdeplot import fast_kde

class DecisionsEns():
    def __init__(self):
        self.decisionfunctionssig = []
        self.decisionfunctionsnoise = []
        self.sigdfvs = []
        self.sigpmfs = []
        self.noisedfvs = []
        self.noisepmfs = []
        self.pdets = []
        self.pfas = []

    def finalize(self):
        self.globalmin = np.floor(np.min(self.decisionfunctionsnoise))
        self.globalmax = np.ceil(np.max(self.decisionfunctionssig))
        self.decfuncx = np.linspace(self.globalmin, self.globalmax, self.globalmax - self.globalmin)
        for ndx in range(len(self.decisionfunctionssig)):
            sigdfv, sigpmf = self.decisionfunctionpmf(self.decisionfunctionssig[ndx])
            self.sigdfvs.append(sigdfv)
            self.sigpmfs.append(sigpmf)
            noisedfv, noisepmf = self.decisionfunctionpmf(self.decisionfunctionsnoise[ndx])
            self.noisedfvs.append(noisedfv)
            self.noisepmfs.append(noisepmf)
            pdet = np.zeros(len(sigpmf))
            pfa = np.zeros(len(noisepmf))
            lastndx = len(sigpmf) - 1
            for ndx in range(lastndx + 1):
                if ndx == 0:
                    pdet[ndx] = sigpmf[lastndx]
                    pfa[ndx] = noisepmf[lastndx]
                else:
                    pdet[ndx] = pdet[ndx - 1] + sigpmf[lastndx - ndx]
                    pfa[ndx] = pfa[ndx - 1] + noisepmf[lastndx - ndx]
            self.pdets.append(pdet)
            self.pfas.append(pfa)
        pass

    def decisionfunctionpmf(self, vals):
        vals = np.append(vals, [self.globalmin, self.globalmax])
        pmf, min, max = fast_kde(vals, bw=4.5)
        x = np.linspace(min, max, len(pmf))
        pmf = pmf / sum(pmf)
        return x, pmf

    def plot_roc(self):
        plt.figure()
        for ndx in range(len(self.pdets)):
            plt.plot(self.pfas[ndx], self.pdets[ndx], 'g', alpha=.1)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel('probability false-alarm')
        plt.ylabel('probability detection')

    def plot_decisionfunctions(self):
        plt.figure()
        for ndx in range(len(self.decisionfunctionssig)):
            plt.plot(self.sigdfvs[ndx], self.sigpmfs[ndx], 'g', alpha=.1)
            plt.plot(self.noisedfvs[ndx], self.noisepmfs[ndx], 'b', alpha=.1)
        sigx, sigpmf = self.decisionfunctionpmf(self.decisionfunctionssig)
        noisex, noisepmf = self.decisionfunctionpmf(self.decisionfunctionsnoise)
        plt.plot(sigx, sigpmf, 'g')
        plt.plot(noisex, noisepmf, 'b')
        plt.xlabel('decision function output-value')
        plt.ylabel('p')

    def addsig(self, decisionfunction):
        self.decisionfunctionssig.append(decisionfunction)

    def addnoise(self, decisionfunction):
        self.decisionfunctionsnoise.append(decisionfunction)

class Decisions():
    def __init__(self, mode, tracker):
        self.mode = mode
        tracker.innov.finalize()
        self.t = tracker.innov.log[:, 0]
        self.yhat = tracker.innov.log[:, 2]
        self.inn = tracker.innov.log[:, 4]
        self.Rhatee = tracker.innov.log[:, 5]
        self.decisionfunction = self.evaldecisionfunction()

    def evaldecisionfunction(self):
        decisionfunction = np.zeros(len(self.inn))
        for ndx in range(len(decisionfunction)):
            delta = (self.yhat[ndx]**2 / 8) - (self.inn[ndx]**2 / (2 * self.Rhatee[ndx]))
            if ndx == 0: decisionfunction[0] = delta
            else: decisionfunction[ndx] = decisionfunction[ndx-1] + delta
        return decisionfunction

def rccircuit(runs):
    from classical import Classical
    ens = DecisionsEns()
    for runndx in range(runs):
        tracker = Classical(mode='rccircuit', plot=False, signal=300.)
        dec = Decisions(mode='rccircuit', tracker=tracker)
        ens.addsig(decisionfunction=dec.decisionfunction)
        tracker = Classical(mode='rccircuit', plot=False, signal=0.)
        dec = Decisions(mode='rccircuit', tracker=tracker)
        ens.addnoise(decisionfunction=dec.decisionfunction)
    ens.finalize()
    return ens

if __name__ == "__main__":
    ens = rccircuit(runs=100)
    ens.plot_decisionfunctions()
    ens.plot_roc()
    plt.show()
    pass