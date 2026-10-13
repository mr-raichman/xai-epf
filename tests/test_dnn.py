"""La xarxa de punt tal com la construeix el codi de Lago et al. (2021): el que s'hi pot
trencar en silenci.

Cap d'aquestes coses no peta quan falla. Una setmana partida entre entrenament i
validació deixa l'aturada mirant dies veïns dels que la xarxa ja ha vist; un escalador
ajustat amb les files de validació hi filtra el que l'aturada hauria de jutjar; un
escalat que no es desfà bé dona preus plausibles i equivocats; i una regla d'aturada
diferent de la del codi entrena més o menys del que toca. Totes donen un MAE que es
pot escriure.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS
from epf.models.dnn.dnn import SCALINGS, fit, fit_rows, improved, inputs
from epf.models.search import split
from epf.parameters import DAYS_PER_WEEK, FOUR_YEARS

# Una xarxa petita i un entrenament curt: el que es comprova són els escaladors, que no
# depenen de com d'ajustada estigui.
TUNED = {"blocks": ["x"], "hidden": [50], "activation": "relu", "l1": 0.0,
         "dropout": 0.0, "batch_normalisation": False,
         "initialiser": "glorot_uniform", "learning_rate": 0.01,
         "scale_inputs": "Median", "scale_outputs": "Median", "seed": 1}


def weeks_of(order):
    """La setmana de cada fila, comptant de set en set des de la primera."""
    return np.asarray(order) // DAYS_PER_WEEK


def test_the_shuffle_moves_whole_weeks():
    # El codi de l'article barreja per setmanes "to avoid data contamination": dos dies
    # veïns, un a cada banda, es mirarien l'un a l'altre.
    train, validate = split(FOUR_YEARS, np.random.default_rng(0))
    order = np.concatenate([train, validate])
    runs = np.split(order, np.flatnonzero(np.diff(weeks_of(order))) + 1)
    assert all(np.array_equal(run, np.arange(run[0], run[0] + len(run))) for run in runs)
    assert len(runs) == FOUR_YEARS // DAYS_PER_WEEK
    assert not np.array_equal(order, np.arange(FOUR_YEARS))


def test_validation_is_the_last_quarter_of_the_rows():
    train, validate = split(FOUR_YEARS, np.random.default_rng(0))
    assert len(validate) == int(0.25 * FOUR_YEARS)
    assert sorted([*train, *validate]) == list(range(FOUR_YEARS))


def test_without_a_generator_the_validation_is_the_calendars_last_quarter():
    # La cerca valida en ordre de calendari.
    train, validate = split(FOUR_YEARS)
    assert list(validate) == list(range(FOUR_YEARS - len(validate), FOUR_YEARS))
    assert list(train) == list(range(FOUR_YEARS - len(validate)))


def test_the_scalers_are_fitted_on_the_training_rows_alone():
    # Abans d'aquesta xarxa, els escaladors s'ajustaven amb tota la finestra, les files
    # de validació incloses. Aquí les de validació porten valors que mourien la mediana.
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 3)), 50 + 10 * rng.normal(size=(120, 24))
    X[90:] += 100.0
    y[90:] += 500.0
    fitted = fit_rows((X[:90], y[:90]), (X[90:], y[90:]), TUNED, seed=1, epochs=2)
    assert np.allclose(fitted.inputs.median, np.median(X[:90], axis=0))
    assert np.allclose(fitted.outputs.median, np.median(y[:90], axis=0))


@pytest.mark.parametrize("name", sorted(SCALINGS))
def test_each_scaling_returns_the_prices_it_was_given(name):
    # Preus com els d'aquest mercat: negatius, zeros i puntes de centenars d'euros.
    rng = np.random.default_rng(0)
    prices = np.concatenate([rng.normal(80, 30, size=(200, 24)),
                             np.full((5, 24), -2.0), np.zeros((5, 24)),
                             rng.uniform(300, 700, size=(5, 24))])
    scaler = SCALINGS[name](prices)
    assert np.allclose(scaler.back(scaler.forward(prices)), prices)


def test_an_epoch_counts_if_the_loss_or_the_error_improves():
    # La regla del codi de l'article, on la de Keras mira només la pèrdua.
    best = (1.0, 10.0)
    assert improved(best, 0.9, 12.0) == (0.9, 12.0)
    assert improved(best, 1.1, 9.0) == (1.0, 9.0)
    assert improved(best, 1.1, 11.0) is None


def test_the_day_is_one_column_from_monday_zero_to_sunday_six():
    days = pd.DatetimeIndex([datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
                             for day in range(DAYS_PER_WEEK)])
    X = pd.DataFrame(np.eye(DAYS_PER_WEEK)[days.dayofweek], index=days,
                     columns=DAY_OF_WEEK_COLUMNS)
    X[[f"lag1_{hour}" for hour in HOUR_COLUMNS]] = 1.0
    values = inputs(X, ["day_of_week", "lag1"])
    assert values.shape == (DAYS_PER_WEEK, 1 + len(HOUR_COLUMNS))
    assert list(values[:, 0]) == list(days.dayofweek)


def test_each_member_fits_on_its_own_numbers():
    # A l'article cada membre surt de la seva cerca, amb els seus blocs i la seva xarxa.
    # Si tots prenguessin els del primer, la mitjana seria d'una xarxa repetida i el
    # pronòstic sortiria igual de plausible.
    days = pd.date_range("2024-01-01", periods=130, freq="D")
    rng = np.random.default_rng(0)
    columns = [f"{block}_{hour}" for block in ("lag1", "lag2") for hour in HOUR_COLUMNS]
    X = pd.DataFrame(rng.normal(size=(len(days), len(columns))), index=days,
                     columns=columns)
    y = pd.DataFrame(50 + 10 * rng.normal(size=(len(days), 24)), index=days,
                     columns=HOUR_COLUMNS)
    members = [TUNED | {"blocks": ["lag1"], "seed": 1},
               TUNED | {"blocks": ["lag1", "lag2"], "seed": 2}]
    model = fit(X, y, days[-1], members, window=120, epochs=2)
    assert [member.blocks for member in model] == [["lag1"], ["lag1", "lag2"]]
    assert [member.fitted.network.input_shape[-1] for member in model] == [24, 48]
