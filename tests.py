import sys; sys.path.extend(['./', './processors', './models'])
from processors.kalman import Kalman
from processors.sigmapoint import SigmaPoint
from processors.particle import Particle
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly
from main import cases

def runner(case):
    def run():
        model, processor = cases[case]['model'], cases[case]['processor']
        processor.run(model)
    return run

for case in cases.keys():
    vars()['test_%s' % case] = runner(case)


