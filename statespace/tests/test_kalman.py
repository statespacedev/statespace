from statespace.threestate import Threestate
from statespace.processors.kalman import Kalman


def test_basic():
    model, processor = Threestate(), Kalman()
    result = processor.run(model)
    assert result
