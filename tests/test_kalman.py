from statespace.extended_kalman_filter import Kalman
from statespace.three_state import Threestate


def test_basic():
    model, processor = Threestate(), Kalman()
    result = processor.run(model)
    assert result
