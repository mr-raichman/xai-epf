"""Alineació i absència de filtració del model LEAR, i els retards de cada finestra."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.models.lear.lear_hyperparameters import EIGHT_WEEKS, TWELVE_WEEKS
from epf.parameters import LAGS
from epf.calendar import HOUR_COLUMNS
from epf.matrix import build
from epf.models.lear.lear import NAME, feature_columns, fit, lags_of, predict

# Prou iteracions perquè els ajusts d'aquestes proves convergeixin; el valor que
# fan servir les execucions és a la configuració.
MAX_ITER = 10000

# Preus sintètics amb un patró setmanal exacte: cada dia de la setmana té sempre la
# mateixa corba. Qualsevol model alineat l'ha de reproduir.
WEEKLY_SHAPE = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]


def forecast(X, y, day, window, max_iter):
    """El que fa el camí endavant amb una sola finestra: un ajust i la seva predicció.

    Amb una finestra la mitjana del conjunt és aquella finestra.
    """
    return predict(fit(X, y, day, windows=(window,), max_iter=max_iter), X, day)[NAME]


def weekly_prices(days, start=datetime.date(2014, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_a_weekly_pattern_is_reproduced():
    X, y = build(weekly_prices(200), LAGS)
    day = X.index[-1]
    predicted = forecast(X, y, day, window=None, max_iter=MAX_ITER)
    assert np.allclose(predicted, y.loc[day].to_numpy(), atol=1.0)


def test_the_day_being_predicted_is_never_read():
    X, y = build(weekly_prices(200), LAGS)
    day = X.index[-1]
    before = forecast(X, y, day, window=None, max_iter=MAX_ITER)

    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    after = forecast(X, corrupted, day, window=None, max_iter=MAX_ITER)
    assert np.array_equal(before, after)


def test_too_short_a_window_fails_loudly():
    X, y = build(weekly_prices(200), LAGS)
    with pytest.raises(ValueError):
        forecast(X, y, X.index[-1], window=50, max_iter=MAX_ITER)


def test_each_short_window_takes_the_lags_that_fit():
    assert lags_of(EIGHT_WEEKS) == (1,)
    assert lags_of(TWELVE_WEEKS) == (1, 2)
    assert lags_of(None) == LAGS
    for window in (EIGHT_WEEKS, TWELVE_WEEKS):
        # LassoLarsIC demana com a mínim dues files més que columnes.
        assert window >= len(feature_columns(lags_of(window))) + 2
