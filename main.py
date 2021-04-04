import sys; sys.path.extend(['./', './processors', './models'])
from processors.kalman import Kalman
from processors.sigmapoint import SigmaPoint
from processors.particle import Particle
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly

cases ={
    '1a': {'model': Onestate(), 'processor': Kalman('ekf')},
    '1b': {'model': Threestate(), 'processor': Kalman('ekf')},
    '1c': {'model': BearingsOnly(), 'processor': Kalman('ekf')},
}
case = '1c'

def main():
    model, processor = cases[case]['model'], cases[case]['processor']
    processor.run(model)
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()


