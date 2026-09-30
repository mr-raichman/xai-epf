"""Alineació i absència de filtració del bosc aleatori."""
import numpy as np
import pytest

from epf.features.matrix import build
from epf.models.forest import fit, forecast
from tests.test_lear import weekly_prices

# Pocs arbres: aquí es comprova l'alineació, no la precisió.
FEW = 20


def test_it_predicts_twenty_four_hours():
    X, y = build(weekly_prices(120))
    out = forecast(X, y, X.index[-1], trees=FEW)
    assert out.shape == (24,)


def test_a_weekly_pattern_is_reproduced():
    X, y = build(weekly_prices(200))
    day = X.index[-1]
    assert np.allclose(forecast(X, y, day, trees=FEW), y.loc[day].to_numpy(), atol=1.0)


def test_the_day_being_predicted_is_never_read():
    X, y = build(weekly_prices(120))
    day = X.index[-1]
    before = forecast(X, y, day, trees=FEW)
    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    assert np.array_equal(before, forecast(X, corrupted, day, trees=FEW))


def test_the_window_limits_the_days_it_sees():
    X, y = build(weekly_prices(200))
    forest = fit(X, y, X.index[-1], window=60, trees=FEW)
    assert forest.n_outputs_ == 24


def test_no_earlier_day_fails_loudly():
    X, y = build(weekly_prices(200))
    with pytest.raises(ValueError):
        forecast(X, y, X.index[0], trees=FEW)
