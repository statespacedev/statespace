import sys
sys.path.append('./')
sys.path.append('./statespace')
sys.path.append('./models')
sys.path.append('./cmake-build-debug/libstatespace')

from statespace.classical import Classical
from models.jazwinski2 import Jazwinski2

processor, model = Classical(), Jazwinski2()
processor.ekfudcpp(model)
processor.innov.plot()

