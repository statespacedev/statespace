import numpy as np
import matplotlib.pyplot as plt


class DecisionsEns():
    def __init__(self, title):
        self.title = title

    def add(self, detector):
        pass

class Rccircuit():
    def __init__(self, tracker):
        self.tracker = tracker

def rccircuit(runs, signal, title=None):
    from classical import Classical
    ens = DecisionsEns(title)
    for runndx in range(runs):
        tracker = Classical('rccircuit', plot=False)
        detector = Rccircuit(tracker)
        ens.add(detector)
    return ens

if __name__ == "__main__":
    enssig = rccircuit(runs=10, signal=300., title='signal')
    # ensnoise = rccircuit(runs=10, signal=0., title='noise')
