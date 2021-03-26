import sys
sys.path.append('./')
sys.path.append('./statespace')
sys.path.append('./models')
# sys.path.append('./cmake-build-debug/libstatespace')

from statespace.classical import Classical
from models.onestate import Onestate

processor, model = Classical(), Onestate()
processor.ekfud(model)
processor.innovs.plot()

