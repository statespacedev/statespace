import sys
sys.path.append('./')
sys.path.append('./statespace')
sys.path.append('./models')
sys.path.append('./cmake-build-debug/libstatespace')

from statespace.classical import Classical
from models.jazwinski2 import Nonlinear2

processor, model = Classical(), Nonlinear2()
processor.ekfudcpp(model)
processor.innovs.plot()

