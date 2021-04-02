import sys
sys.path.append('./')
sys.path.append('./processors')
sys.path.append('./models')

def main():
    # onestate_ekf()
    # threestate_ekf()
    bearingsonly_ekf()

def onestate_ekf():
    from models.onestate import Onestate
    from processors.kalman import Kalman
    model = Onestate()
    processor = Kalman()
    processor.ekf(model)
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

def threestate_ekf():
    from models.threestate import Threestate
    from processors.kalman import Kalman
    model = Threestate()
    processor = Kalman()
    processor.ekf(model)
    model.eval.estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.show()

def bearingsonly_ekf():
    from models.bearingsonly import BearingsOnly
    from processors.kalman import Kalman
    model = BearingsOnly()
    processor = Kalman()
    processor.ekf(model)
    # model.eval.model()
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()

if __name__ == "__main__":
    main()


