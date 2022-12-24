from statespace.processors.extended_kalman_filter import Kalman
from statespace.processors.sigma_point_kalman_filter import SigmaPoint
from statespace.processors.particle_filter import Particle
from statespace.models.one_state import Onestate
from statespace.models.three_state import Threestate
from statespace.models.bearings_only import BearingsOnly

cases = {
    '1a': {'model': Onestate(), 'processor': Kalman()},
    '1b': {'model': Threestate(), 'processor': Kalman()},
    '1c': {'model': Onestate(), 'processor': Kalman('ud')},
    '1d': {'model': Threestate(), 'processor': Kalman('ud')},
    '2a': {'model': Onestate(), 'processor': SigmaPoint()},
    '2b': {'model': Threestate(), 'processor': SigmaPoint()},
    '2c': {'model': Onestate(), 'processor': SigmaPoint('cholesky')},
    '2d': {'model': Threestate(), 'processor': SigmaPoint('cholesky')},
    '3a': {'model': Onestate(), 'processor': Particle()},
    '3b': {'model': Threestate(), 'processor': Particle()},
    'bo1': {'model': BearingsOnly(), 'processor': Kalman()},
    'bo2': {'model': BearingsOnly(), 'processor': SigmaPoint()},  # tuning
    'bo3': {'model': BearingsOnly(), 'processor': Particle()},  # tuning
}
case = '3b'


def main():
    model, processor = cases[case]['model'], cases[case]['processor']
    processor.run(model)
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()


if __name__ == "__main__":
    main()
