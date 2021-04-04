import sys; sys.path.extend(['./', './processors', './models'])
from processors.kalman import Kalman
from processors.sigmapoint import SigmaPoint
from processors.particle import Particle
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly

cases =[[0],
        [1, Onestate(), Kalman('ekf')],
        [2, Threestate(), Kalman('ekf')],
        [3, BearingsOnly(), Kalman('ekf')],
        ]
case = 3
def main():
    model, processor = cases[case][1], cases[case][2]
    processor.run(model)
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()


