"""Aïllament del dia que es prediu i que les xarxes del conjunt siguin diferents."""
import datetime

import numpy as np
import pandas as pd

from epf.data.calendar import HOUR_COLUMNS
from epf.features.matrix import build
from epf.models.mlp import fit, forecast

# Preus sintètics amb un patró setmanal exacte, com als altres models.
WEEKLY_SHAPE = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]

SMALL = {"hidden": (8,), "members": 2}


def daily_prices(days, start=datetime.date(2014, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return build(pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS))


def test_the_day_being_predicted_is_never_read():
    X, y = daily_prices(200)
    day = X.index[-1]
    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    before = forecast(X, y, day, window=100, **SMALL)
    assert np.array_equal(before, forecast(X, corrupted, day, window=100, **SMALL))


def test_nothing_after_the_predicted_day_reaches_the_scalers():
    # Els escaladors surten de la finestra de calibratge. Si s'ajustessin sobre tota
    # la matriu, un dia posterior en mouria la mitjana i la desviació, i amb elles la
    # predicció, sense que res fallés.
    X, y = daily_prices(200)
    day = X.index[-20]
    later = X.copy()
    later.loc[X.index[-1]] = 9999.0
    assert np.array_equal(forecast(X, y, day, window=100, **SMALL),
                          forecast(later, y, day, window=100, **SMALL))


def test_the_members_of_the_ensemble_are_different_networks():
    # Amb la mateixa llavor a totes, la mitjana de quatre xarxes seria una xarxa i el
    # conjunt no voldria dir res.
    X, y = daily_prices(200)
    day = X.index[-1]
    networks, _, _ = fit(X, y, day, window=100, hidden=(8,), members=3)
    first = networks[0].coefs_[0]
    assert all(not np.array_equal(first, other.coefs_[0]) for other in networks[1:])
    assert not np.allclose(forecast(X, y, day, window=100, hidden=(8,), members=1),
                           forecast(X, y, day, window=100, hidden=(8,), members=3))


def test_it_predicts_the_twenty_four_hours_in_order():
    X, y = daily_prices(300)
    prices = forecast(X, y, X.index[-1], window=200, hidden=(32, 32), members=2)
    assert prices.shape == (24,)
    # El preu sintètic creix amb l'hora i la xarxa ha de recollir-ne el pendent.
    assert np.corrcoef(prices, np.arange(24))[0, 1] > 0.9
