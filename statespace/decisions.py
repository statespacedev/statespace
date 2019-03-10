import numpy as np
import matplotlib.pyplot as plt


class DecisionsEns():
    def __init__(self):
        self.decisionfunctionssig = []
        self.decisionfunctionsnoise = []
        self.rocs = []

    def addsig(self, decisionfunction):
        self.decisionfunctionssig.append(decisionfunction)

    def addnoise(self, decisionfunction):
        self.decisionfunctionsnoise.append(decisionfunction)

    def plot_decisionfunctions(self):
        from pymc3.plots.kdeplot import fast_kde
        import matplotlib.pyplot as plt
        globalmin = np.floor(np.min(self.decisionfunctionsnoise))
        globalmax = np.ceil(np.max(self.decisionfunctionssig))
        def pmf(vals):
            vals = np.append(vals, [globalmin, globalmax])
            pmf, min, max = fast_kde(vals, bw=4.5)
            x = np.linspace(min, max, len(pmf))
            pmf = pmf / sum(pmf)
            return x, pmf
        plt.figure()
        for ndx in range(len(self.decisionfunctionssig)):
            sigx, sigpmf = pmf(self.decisionfunctionssig[ndx])
            noisex, noisepmf = pmf(self.decisionfunctionsnoise[ndx])
            plt.plot(sigx, sigpmf, 'g', alpha=.1)
            plt.plot(noisex, noisepmf, 'b', alpha=.1)
        sigx, sigpmf = pmf(self.decisionfunctionssig)
        noisex, noisepmf = pmf(self.decisionfunctionsnoise)
        plt.plot(sigx, sigpmf, 'g')
        plt.plot(noisex, noisepmf, 'b')
        plt.xlabel('decision function output-value')
        plt.ylabel('p')

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
    return ens

if __name__ == "__main__":
    ens = rccircuit(runs=100)
    ens.plot_decisionfunctions()
    plt.show()
    pass