import sys
sys.path.append('./')
sys.path.append('./processors')
sys.path.append('./models')

def main():
    onestate_ekf()
    # bearingsonly_ekf()

def bearingsonly_ekf():
    from models.bearingsonly import BearingsOnly
    from processors.classical import Classical
    model = BearingsOnly()
    processor = Classical()
    processor.ekf(model)
    model.eval.plot_model()
    model.eval.plot_estimate(processor.log)
    model.eval.show()

def onestate_ekf():
    from models.onestate import Onestate
    from processors.classical import Classical
    model = Onestate()
    processor = Classical()
    processor.ekf(model)
    model.eval.plot_estimate(processor.log)
    model.eval.autocorr.run(processor.log)
    model.eval.autocorr.plot()
    model.eval.show()

if __name__ == "__main__":
    main()