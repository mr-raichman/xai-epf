import numpy as np
import pytest

from epf import Refusal
from epf.contract.metrics import scored


def test_a_score_refuses_a_prediction_that_is_not_a_number_and_fails_a_missing_price():
    with pytest.raises(Refusal, match="la predicció té 2 valors no finits"):
        scored([1.0, np.nan, np.inf], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="el preu real té 1 valor absent") as raised:
        scored([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
    assert not isinstance(raised.value, Refusal)
