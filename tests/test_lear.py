"""Alineació, absència de filtració i transformació del model LEAR."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.models.lear.lear_hyperparameters import EIGHT_WEEKS, TWELVE_WEEKS
from epf.parameters import LAGS
from epf.calendar import HOUR_COLUMNS
from epf.matrix import build
from epf.attribution.stored import TRANSFORMED_TOLERANCE
from epf.models.lear.lear import (DESIGN_COLUMNS, NAME, Scaler, attribute,
                                  design_columns, fit, lags_of, predict,
                                  week_dummies)

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


def explained(X, y, day, windows, max_iter):
    """L'atribució del mateix ajust que hauria predit el dia."""
    return attribute(fit(X, y, day, windows=windows, max_iter=max_iter), X, day)


def weekly_prices(days, start=datetime.date(2014, 1, 1)):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_the_scaler_round_trips():
    values = np.array([[0.0, 5.0], [100.0, -3.0]])
    scaler = Scaler(values)
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_each_column_gets_its_own_median_and_spread():
    # Una columna al voltant de 0 i una altra al voltant de 1000. Amb una mediana
    # compartida, la primera columna quedaria tota desplaçada cap avall.
    values = np.array([[0.0, 1000.0], [1.0, 1002.0], [2.0, 1004.0], [3.0, 1006.0]])
    scaler = Scaler(values)
    assert list(scaler.median) == [1.5, 1003.0]
    transformed = scaler.forward(values)
    assert abs(transformed[:, 0].mean()) < 1e-9
    assert abs(transformed[:, 1].mean()) < 1e-9
    assert np.allclose(scaler.back(transformed), values)


def test_a_window_without_spread_fails_loudly():
    with pytest.raises(ValueError):
        Scaler(np.zeros((10, 24)))


def test_week_dummies_are_one_hot_on_the_right_day():
    dummies = week_dummies([0, 6, 3])
    assert dummies.shape == (3, 7)
    assert list(dummies.sum(axis=1)) == [1, 1, 1]
    assert dummies[1][6] == 1 and dummies[2][3] == 1


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


def noisy_prices(days, start=datetime.date(2014, 1, 1)):
    """El mateix patró setmanal amb soroll, perquè cap coeficient no sigui zero."""
    price = weekly_prices(days, start)
    noise = np.random.default_rng(0).normal(0.0, 2.0, size=price.shape)
    return price + noise


def test_each_short_window_takes_the_lags_that_fit():
    assert lags_of(EIGHT_WEEKS) == (1,)
    assert lags_of(TWELVE_WEEKS) == (1, 2)
    assert lags_of(None) == LAGS
    for window in (EIGHT_WEEKS, TWELVE_WEEKS):
        # LassoLarsIC demana com a mínim dues files més que columnes.
        assert window >= len(design_columns(lags_of(window))) + 2


def test_a_short_window_leaves_the_lags_it_cannot_carry_at_zero():
    X, y = build(noisy_prices(200), LAGS)
    attribution = explained(X, y, X.index[-1], windows=(EIGHT_WEEKS,), max_iter=MAX_ITER)
    carried = set(design_columns(lags_of(EIGHT_WEEKS)))
    absent = [number for number, name in enumerate(DESIGN_COLUMNS) if name not in carried]
    assert np.count_nonzero(attribution.values[0][absent]) == 0
    assert np.count_nonzero(attribution.values[0]) > 0


def test_the_base_and_the_contributions_return_the_members_own_prediction():
    # L'additivitat que justifica desar la base, i l'única comprovació possible per a
    # LEAR: la suma és a l'espai transformat i `sinh` no deixa comparar-la amb el CSV.
    # Mentre la predicció transformada no es desava, la base de cada finestra no la
    # comprovava res.
    # Dues finestres i no una, perquè el que es pot trencar en silenci és
    # l'alineació: la base del membre m contra la predicció del membre m.
    X, y = build(noisy_prices(200), LAGS)
    one = explained(X, y, X.index[-1], windows=(EIGHT_WEEKS, TWELVE_WEEKS),
                    max_iter=MAX_ITER)
    rebuilt = one.base + one.values.sum(axis=1)
    assert rebuilt.shape == one.transformed.shape
    assert np.abs(rebuilt - one.transformed).max() < TRANSFORMED_TOLERANCE


def test_moving_a_column_of_the_predicted_day_moves_only_its_own_contribution():
    # Les contribucions són el coeficient per la distància de la columna a la mitjana
    # de la finestra, i la finestra no conté el dia que es prediu.
    X, y = build(noisy_prices(400), LAGS)
    day = X.index[-1]
    before = explained(X, y, day, windows=(200,), max_iter=MAX_ITER)
    shifted = X.copy()
    shifted.loc[day, "lag1_h05"] += 5.0
    after = explained(shifted, y, day, windows=(200,), max_iter=MAX_ITER)

    moved = after.values[0] - before.values[0]
    row = DESIGN_COLUMNS.index("lag1_h05")
    assert np.abs(moved[row]).max() > 0
    assert np.count_nonzero(np.delete(moved, row, axis=0)) == 0
