from statespace.processors.extended_kalman_filter import Kalman
from statespace.models.three_state import Threestate


def test_basic():
    model, processor = Threestate(), Kalman()
    result = processor.run(model)
    assert result
