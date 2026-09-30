"""Alineació, absència de filtració i transformació del model LEAR."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.data.calendar import HOUR_COLUMNS
from epf.features.matrix import build
from epf.models.lear import Scaler, forecast, week_dummies

# Preus sintètics amb un patró setmanal exacte: cada dia de la setmana té sempre la
# mateixa corba. Qualsevol model alineat l'ha de reproduir.
WEEKLY_SHAPE = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]


def weekly_prices(days, start=datetime.date(2014, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_the_scaler_round_trips():
    values = np.array([[0.0, 5.0], [100.0, -3.0]])
    scaler = Scaler(values)
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_a_window_without_spread_fails_loudly():
    with pytest.raises(ValueError):
        Scaler(np.zeros((10, 24)))


def test_week_dummies_are_one_hot_on_the_right_day():
    dummies = week_dummies([0, 6, 3])
    assert dummies.shape == (3, 7)
    assert list(dummies.sum(axis=1)) == [1, 1, 1]
    assert dummies[1][6] == 1 and dummies[2][3] == 1


def test_a_weekly_pattern_is_reproduced():
    X, y = build(weekly_prices(200))
    day = X.index[-1]
    predicted = forecast(X, y, day, window=None)
    assert np.allclose(predicted, y.loc[day].to_numpy(), atol=1.0)


def test_the_day_being_predicted_is_never_read():
    X, y = build(weekly_prices(200))
    day = X.index[-1]
    before = forecast(X, y, day, window=None)

    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    after = forecast(X, corrupted, day, window=None)
    assert np.array_equal(before, after)


def test_too_short_a_window_fails_loudly():
    X, y = build(weekly_prices(200))
    with pytest.raises(ValueError):
        forecast(X, y, X.index[-1], window=50)
