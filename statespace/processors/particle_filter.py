"""particle filters, aka sequential monte carlo sampling processors. sampling here is random, not deterministic as in
the sigma point kalman filter - and the idea of resampling and evolving newer, fitter particles comes to the fore.
the particles are generated randomly and new ones are introduced freely, but generally guided by our evolving beliefs
around our model and external reality - we're at the least in the space of the 'genetic algorithms' -
survival-of-the-fittest is a core concept here, but the 'genetic material' concept is demphasized in favor of using
sensors to bring in new additional data from the external world over time. everything here is about evolution of
beliefs over time, rather than static optimizations at a single point in time - sensor-based closed-loop feedback
estimation and control.

the particles live, die, and new ones are born via 'resampling' - this is survival-of-the-fittest. in fact,
new ones essentialy kill old ones - things are quite red of tooth and claw. basic idea is - when new data comes in,
some of the particles agree with the data better than other particles - the former are fitter, the latter are less
fit. 'resample' the less fit particles - replace them with new random particles that are more fit - that agree with
the data better - then life goes on until the next new data comes in, the endless cycle repeats - evolve, update,
evolve, update.

we can see right away how particles can soak up a completely arbitrary amount of hardware flops - just keep adding
more particles until your hardware gives up the ghost. for a state vector with x components, spkf gives us 2x + 1
particles - two sigma points/particles for each state component, and one particle representing the estimated mean
state. with sequential monte carlo, we now simply have a 'cloud' of particles swarming about over time - at any time
we can measure the cloud's estimated mean state using kernal density methods, likewise for the shape or scatter of
the cloud and various measures of uncertainty."""
import numpy as np


class Particle:
    """particle filter, sequential monte carlo processor"""

    def __init__(self, conf):
        self.conf, self.log = conf, []
        self.run = self.run_model

    def run_model(self, model):
        """basic sampling-importance-resampling particle filter, bootstrap particle filter, condensation particle
        filter, survival of the fittest algorithm"""
        sim, _, h = model.entities()
        X, predict, update, resample = model.pf.entities()
        for t, o, u in sim():
            X = predict(X, u)
            W = update(X, o)
            x = np.sum(np.multiply(W, X), axis=1).reshape(-1, 1)
            X = resample(X, W)
            self.log.append([t, x, h(x)])


if __name__ == "__main__":
    pass
