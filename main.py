import sys; sys.path.extend(['./', './processors', './models'])
from processors.kalman import Kalman
from processors.sigmapoint import SigmaPoint
from processors.particle import Particle
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly

cases ={
    '1a': {'model': Onestate(), 'processor': Kalman()},
    '1b': {'model': Threestate(), 'processor': Kalman()}, # tuning
    '1c': {'model': BearingsOnly(), 'processor': Kalman()}, # tuning
    '2a': {'model': Onestate(), 'processor': Kalman('ud')},
    '2b': {'model': Threestate(), 'processor': Kalman('ud')},
    '2c': {'model': BearingsOnly(), 'processor': Kalman('ud')}, # tuning
    '4a': {'model': Onestate(), 'processor': SigmaPoint()}, # todo
    '4b': {'model': Threestate(), 'processor': SigmaPoint()}, # todo
    '4c': {'model': BearingsOnly(), 'processor': SigmaPoint()}, # todo
    '5a': {'model': Onestate(), 'processor': SigmaPoint('cho')}, # todo
    '5b': {'model': Threestate(), 'processor': SigmaPoint('cho')}, # todo
    '5c': {'model': BearingsOnly(), 'processor': SigmaPoint('cho')}, # todo
}
case = '2c'

def main():
    model, processor = cases[case]['model'], cases[case]['processor']
    processor.run(model)
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()

