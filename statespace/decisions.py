import numpy as np
import matplotlib.pyplot as plt


class DecisionsEns():
    def __init__(self, title):
        self.title = title
        self.decisionfunctions = []

    def update(self, decisionfunction):
        self.decisionfunctions.append(decisionfunction)

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

def rccircuit(runs, signal, title=None):
    from classical import Classical
    ens = DecisionsEns(title)
    for runndx in range(runs):
        tracker = Classical(mode='rccircuit', plot=False, signal=signal)
        dec = Decisions(mode='rccircuit', tracker=tracker)
        ens.update(decisionfunction=dec.decisionfunction)
    return ens

def plot_ensemble_decision_function(enssig, ensnoise):
    def pmf(vals):
        pmf, min, max = fast_kde(vals, bw=4.5)
        x = np.linspace(min, max, len(pmf))
        pmf = pmf / sum(pmf)
        return x, pmf
    from pymc3.plots.kdeplot import fast_kde
    import matplotlib.pyplot as plt
    plt.figure()
    for ndx in range(len(enssig.decisionfunctions)):
        sigx, sigpmf = pmf(enssig.decisionfunctions[ndx])
        noisex, noisepmf = pmf(ensnoise.decisionfunctions[ndx])
        plt.plot(sigx, sigpmf, 'g', alpha=.1)
        plt.plot(noisex, noisepmf, 'b', alpha=.1)
    sigx, sigpmf = pmf(enssig.decisionfunctions)
    noisex, noisepmf = pmf(ensnoise.decisionfunctions)
    plt.plot(sigx, sigpmf, 'g')
    plt.plot(noisex, noisepmf, 'b')
    plt.xlabel('decision function value')
    plt.ylabel('p')

if __name__ == "__main__":
    enssig = rccircuit(runs=100, signal=300., title='signal')
    ensnoise = rccircuit(runs=100, signal=0., title='noise')
    plot_ensemble_decision_function(enssig=enssig, ensnoise=ensnoise)
    plt.show()

    pass