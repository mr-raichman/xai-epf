"""Alineació, absència de filtració i criteri del model LEAR."""
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC

from epf.parameters import LAGS
from epf.calendar import HOUR_COLUMNS
from epf.matrix import build, calibration
from epf.models.lear import lear
from epf.models.lear.lear import NAME, fit, predict

# Prou iteracions perquè els ajusts d'aquestes proves convergeixin; el valor que
# fan servir les execucions és a la configuració.
MAX_ITER = 10000

# Dies de calibratge de les proves: les sèries de 200 dies en deixen 193 un cop
# descomptat l'escalfament, i la finestra n'ha de poder omplir tants com en demana.
WINDOW = 150

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
    predicted = forecast(X, y, day, window=WINDOW, max_iter=MAX_ITER)
    assert np.allclose(predicted, y.loc[day].to_numpy(), atol=1.0)


def test_the_day_being_predicted_is_never_read():
    X, y = build(weekly_prices(200), LAGS)
    day = X.index[-1]
    before = forecast(X, y, day, window=WINDOW, max_iter=MAX_ITER)

    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    after = forecast(X, corrupted, day, window=WINDOW, max_iter=MAX_ITER)
    assert np.array_equal(before, after)


def test_the_penalty_takes_the_targets_variance_as_the_noise():
    # El criteri de l'scikit-learn de Lago et al. (2021). Sense, una finestra llarga
    # tria una altra lambda sense queixar-se, i només les curtes petarien.
    days = pd.date_range("2014-01-01", periods=400, freq="D", name="day")
    noise = np.random.default_rng(0).normal(0.0, 2.0, size=(400, 24))
    price = pd.DataFrame([[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)]
                          for day in days], index=days, columns=HOUR_COLUMNS) + noise
    X, y = build(price, LAGS)
    day = X.index[-1]
    (fitted,) = fit(X, y, day, windows=(300,), max_iter=MAX_ITER)
    features, target = calibration(X, y, day, 300)
    design = lear._design(features, fitted.prices)
    scaled = fitted.target.forward(target.to_numpy())
    for hour in (0, 12, 23):
        given = LassoLarsIC(criterion="aic", max_iter=MAX_ITER,
                            noise_variance=np.var(scaled[:, hour]))
        estimated = LassoLarsIC(criterion="aic", max_iter=MAX_ITER)
        assert fitted.models[hour].alpha == given.fit(design, scaled[:, hour]).alpha_
        assert fitted.models[hour].alpha != estimated.fit(design, scaled[:, hour]).alpha_
