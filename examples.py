import sys
sys.path.append('./')
sys.path.append('./processors')
sys.path.append('./models')

from processors.classical import Classical
from models.onestate import Onestate

processor, model = Classical(), Onestate()
processor.ekf(model)
model.eval.plot_estimate(processor.log)

