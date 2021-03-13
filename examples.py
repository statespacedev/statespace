import sys
sys.path.append('./')
sys.path.append('./statespace')
sys.path.append('./models')
sys.path.append('./cmake-build-debug/libstatespace')

from models.jazwinski2 import Jazwinski2
from statespace.classical import Classical

processor, model = Classical(), Jazwinski2()
processor.ekfud(model)
processor.innov.plot()

