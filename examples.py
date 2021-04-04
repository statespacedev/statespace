import sys
sys.path.append('./')
sys.path.append('./processors')
sys.path.append('./models')
from processors.kalman import Kalman
from models.onestate import Onestate
from models.threestate import Threestate
from models.bearingsonly import BearingsOnly

def main():
    # onestate_ekf()
    # threestate_ekf()
    bearingsonly_ekf()

def onestate_ekf():
    model = Onestate()
    processor = Kalman(model, 'ekf')
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

def threestate_ekf():
    model = Threestate()
    processor = Kalman(model, 'ekf')
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

def bearingsonly_ekf():
    model = BearingsOnly()
    processor = Kalman(model, 'ekf')
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()


