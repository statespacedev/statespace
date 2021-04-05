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

def runner(case):
    def run():
        model, processor = cases[case]['model'], cases[case]['processor']
        processor.run(model)
    return run

for case in cases.keys():
    vars()['test_%s' % case] = runner(case)


