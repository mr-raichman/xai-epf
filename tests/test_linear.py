"""La xarxa elàstica: cada hora amb les seves files, la solar de nit i dels vespres,
l'atribució que torna la predicció, i el camí endavant en ordre d'hora i sense el dia
que prediu."""
import datetime

import numpy as np
import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.matrix import HOUR, build, forecasts, per_hour
from epf.models.linear.linear import (NAME, attribute_rows, fit, fit_rows, predict,
                                      predict_rows)
from epf.parameters import LAGS

PENALTY, MIXING = 0.01, 0.5

# Hores en què la previsió solar val zero tots els dies, i l'hora del vespre en què val
# zero més de la meitat dels dies: les dues formes que pren a les dades de debò.
NIGHT, EVENING = (0, 1, 2, 3, 4, 5, 23), 20


def per_hour_table(days=120, seed=0):
    """La taula per hora del conjunt de previsions, sobre sèries sintètiques."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D", name="day")

    def frame(values):
        return pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)

    solar = rng.uniform(0, 8000, (days, 24))
    solar[:, NIGHT] = 0.0
    solar[:, EVENING] = np.where(rng.random(days) < 0.6, 0.0, solar[:, EVENING])
    day_ahead = {"demand": frame(rng.uniform(20000, 40000, (days, 24))),
                 "wind": frame(rng.uniform(500, 15000, (days, 24))),
                 "solar": frame(solar)}
    return forecasts(frame(rng.uniform(-2, 200, (days, 24))), day_ahead).hourly


# Preus sintètics amb un patró setmanal exacte, com als altres models.
WEEKLY_SHAPE = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]


def hourly_prices(days, start=datetime.date(2014, 1, 1)):
    """La taula per hora dels retards de preu, sobre un preu que puja un euro cada hora."""
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)],
                             name="day")
    rows = [[WEEKLY_SHAPE[day.dayofweek] + hour for hour in range(24)] for day in index]
    return per_hour(*build(pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS), LAGS))


def test_each_hours_model_reads_only_the_rows_of_its_hour():
    X, y = per_hour_table()
    before = fit_rows(X, y, PENALTY, MIXING)
    changed = y.copy()
    changed[(X[HOUR] == 5).to_numpy()] += 40 * np.arange((X[HOUR] == 5).sum()) % 90
    after = fit_rows(X, changed, PENALTY, MIXING)
    for hour in range(24):
        same = np.array_equal(before[hour].model.coef_, after[hour].model.coef_)
        assert same == (hour != 5)


def test_solar_leaves_the_night_hours_and_keeps_its_evening_without_mad():
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    for hour in NIGHT:
        assert "solar" not in model[hour].columns
    evening = model[EVENING]
    assert "solar" in evening.columns
    solar = X.loc[(X[HOUR] == EVENING).to_numpy(), "solar"].to_numpy()
    assert evening.inputs.scale[evening.columns.index("solar")] == np.std(solar)
    explained = attribute_rows(model, X)
    night = X[HOUR].isin(NIGHT).to_numpy()
    assert (explained.values.loc[night, "solar"] == 0).all()


def test_over_the_rows_it_was_fitted_on_each_contribution_averages_zero():
    # La propietat que en fa valors de SHAP: el centre és la mitjana de les files d'ajust
    # de l'hora, i per això la base és la predicció mitjana d'aquelles files. La suma de
    # sota es compleix amb qualsevol centre i no ho podria veure.
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    explained = attribute_rows(model, X)
    for hour in range(24):
        rows = (X[HOUR] == hour).to_numpy()
        assert np.allclose(explained.values[rows].mean(), 0, atol=1e-9)
        assert np.isclose(explained.base[rows].iloc[0], explained.transformed[rows].mean())


def test_moving_one_column_of_a_row_moves_only_its_own_contribution():
    # L'ordre de la matriu del model, les escalades i després els indicadors, no és el de
    # la taula: una contribució escrita a la columna que no toca sumaria igual i tindria
    # la mateixa mitjana.
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    before = attribute_rows(model, X).values
    shifted = X.copy()
    rows = (shifted[HOUR] == 13).to_numpy()
    shifted.loc[rows, "wind"] += 1000.0
    moved = attribute_rows(model, shifted).values - before
    assert (moved.loc[rows, "wind"] != 0).all()
    assert (moved.drop(columns="wind") == 0).all().all()
    assert (moved.loc[~rows] == 0).all().all()


def test_the_base_and_the_contributions_return_the_rows_own_prediction():
    # Dos camins independents: les contribucions surten de `coef_` i `intercept_`, la
    # predicció de `predict`, i `sinh` torna aquesta al preu que dona el model.
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    explained = attribute_rows(model, X)
    summed = explained.base + explained.values.sum(axis="columns")
    assert np.allclose(summed, explained.transformed, atol=1e-9)
    prices = predict_rows(model, X)
    for hour in range(24):
        rows = (X[HOUR] == hour).to_numpy()
        back = model[hour].target.back(explained.transformed[rows].to_numpy()[:, None])
        assert np.allclose(back.ravel(), prices[rows])


def test_the_walk_predicts_the_twenty_four_hours_in_order():
    # Cada hora té el seu model, i amb aquest preu totes resolen el mateix problema
    # desplaçat un euro: si les hores sortissin barrejades, els salts no serien d'u.
    X, y = hourly_prices(200)
    day = X.index[-1]
    prices = predict(fit(X, y, day, window=100), X, day)[NAME]
    assert prices.shape == (24,)
    assert np.allclose(np.diff(prices), 1.0)


def test_the_day_being_predicted_is_never_read():
    X, y = hourly_prices(200)
    day = X.index[-1]
    corrupted = y.copy()
    corrupted.loc[day] = 9999.0
    before = predict(fit(X, y, day, window=100), X, day)[NAME]
    assert np.array_equal(before, predict(fit(X, corrupted, day, window=100), X, day)[NAME])
