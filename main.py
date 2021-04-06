import sys; sys.path.extend(['./', './processors', './models'])
from processors.kalman import Kalman
from processors.sigmapoint import SigmaPoint
from processors.particle import Particle
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly

cases ={
    '1a': {'model': Onestate(), 'processor': Kalman()},
    '1b': {'model': Onestate(), 'processor': Kalman('ud')},
    '1c': {'model': Onestate(), 'processor': SigmaPoint()}, # tuning
    '1d': {'model': Onestate(), 'processor': SigmaPoint('cho')}, # tuning
    '2a': {'model': Threestate(), 'processor': Kalman()},
    '2b': {'model': Threestate(), 'processor': Kalman('ud')},
    '2c': {'model': Threestate(), 'processor': SigmaPoint()}, # tuning
    '2d': {'model': Threestate(), 'processor': SigmaPoint('cho')},
    '3a': {'model': BearingsOnly(), 'processor': Kalman()}, # tuning
    '3b': {'model': BearingsOnly(), 'processor': Kalman('ud')}, # tuning
    '3c': {'model': BearingsOnly(), 'processor': SigmaPoint()}, # todo
    '3d': {'model': BearingsOnly(), 'processor': SigmaPoint('cho')}, # todo
}
case = '1a'

def main():
    model, processor = cases[case]['model'], cases[case]['processor']
    processor.run(model)
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()

