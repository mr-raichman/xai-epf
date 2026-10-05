"""Alineació, absència de filtració i forma de l'atribució del gradient boosting."""
import datetime
import functools

import numpy as np
import pandas as pd

from epf.data.calendar import HOUR_COLUMNS
from epf.features.matrix import build, per_hour
from epf.models.boosting import explain, forecast, lightgbm

# Preus sintètics amb un patró setmanal exacte, com als altres models.
WEEKLY_SHAPE = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]

SMALL = functools.partial(lightgbm, trees=20)


def hourly_prices(days, start=datetime.date(2014, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return per_hour(*build(pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)))


def test_it_predicts_the_twenty_four_hours_in_order():
    X, y = hourly_prices(200)
    prices = forecast(SMALL, X, y, X.index[-1], window=100)
    assert prices.shape == (24,)
    # El preu sintètic creix amb l'hora; un arbre arrodoneix per fulles, així que
    # el que ha de sortir és una corba que no baixa, no pas 24 valors diferents.
    assert (np.diff(prices) >= 0).all()
    assert prices.max() > prices.min()


def test_the_attribution_is_columns_by_delivery_hour():
    X, y = hourly_prices(200)
    day = X.index[-1]
    prices, attribution = explain(SMALL, X, y, day, window=100)
    assert attribution.values.shape == (1, X.shape[1], 24)
    assert list(attribution.columns) == list(X.columns)
    assert np.array_equal(prices, forecast(SMALL, X, y, day, window=100))


def test_the_day_being_predicted_is_never_read():
    X, y = hourly_prices(200)
    day = X.index[-1]
    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    before = forecast(SMALL, X, y, day, window=100)
    assert np.array_equal(before, forecast(SMALL, X, corrupted, day, window=100))
