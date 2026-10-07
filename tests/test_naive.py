"""Les tres prediccions ingènues prenen l'hora que toca."""
import numpy as np
import pandas as pd

from epf.parameters import LAGS
from epf.matrix import build
from epf.models.naive.naive import previous_day, previous_week, similar_day
from tests.test_matrix import frame_of


def test_each_one_copies_its_own_lag():
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    day = X.index[10]
    assert np.array_equal(previous_day(X, day),
                          price.loc[day - pd.Timedelta(days=1)].to_numpy())
    assert np.array_equal(previous_week(X, day),
                          price.loc[day - pd.Timedelta(days=7)].to_numpy())


def test_the_similar_day_takes_yesterday_only_from_tuesday_to_friday():
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    for day in X.index[:7]:
        expected = 1 if day.dayofweek in (1, 2, 3, 4) else 7
        assert np.array_equal(similar_day(X, day),
                              price.loc[day - pd.Timedelta(days=expected)].to_numpy())


def test_the_weekend_and_monday_reach_a_week_back():
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    weekend = [day for day in X.index if day.dayofweek in (5, 6, 0)]
    for day in weekend:
        assert np.array_equal(similar_day(X, day), previous_week(X, day))
