import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNetCV

from epf import Refusal
from epf.contract.matrix import HOUR, PRICE, matrices
from epf.models.elastic_net.attribution import attribute_rows
from epf.models.elastic_net.regression import _prepared, by_hour, fit_rows, predict_rows
from epf.models.elastic_net.scaling import Scaler
from epf.models.elastic_net.search import penalties
from epf.parameters import HOUR_COLUMNS

PENALTY, MIXING = 0.01, 0.5

# Hores en què la previsió solar val zero tots els dies, i l'hora del vespre en què val
# zero més de la meitat dels dies: les dues formes que pren a les dades de debò.
NIGHT, EVENING = (0, 1, 2, 3, 4, 5, 23), 20


def per_hour_table(days=120, seed=0):
    """Torna la matriu per hora d'unes sèries sintètiques."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=days, freq="D", name="day")

    def frame(values):
        return pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)

    solar = rng.uniform(0, 8000, (days, 24))
    solar[:, NIGHT] = 0.0
    solar[:, EVENING] = np.where(rng.random(days) < 0.6, 0.0, solar[:, EVENING])
    series = {
        "demand_fcst": frame(rng.uniform(20000, 40000, (days, 24))),
        "wind_fcst": frame(rng.uniform(500, 15000, (days, 24))),
        "solar_fcst": frame(solar),
        PRICE: frame(rng.uniform(-2, 200, (days, 24))),
        "nuclear_sched": frame(rng.uniform(4000, 7000, (days, 24))),
        "coal_gas_sched": frame(rng.uniform(2000, 15000, (days, 24))),
        "hydro_sched": frame(rng.uniform(1000, 10000, (days, 24))),
    }
    return matrices(series).hourly


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
        assert "solar_fcst" not in model[hour].columns
    evening = model[EVENING]
    assert "solar_fcst" in evening.columns
    solar = X.loc[(X[HOUR] == EVENING).to_numpy(), "solar_fcst"].to_numpy()
    assert evening.inputs.scale[evening.columns.index("solar_fcst")] == np.std(solar)
    explained = attribute_rows(model, X)
    night = X[HOUR].isin(NIGHT).to_numpy()
    assert (explained.values.loc[night, "solar_fcst"] == 0).all()


def test_over_the_rows_it_was_fitted_on_each_contribution_averages_zero():
    """La propietat que en fa valors de SHAP, que la suma no podria veure.

    El centre és la mitjana de les files d'ajust de l'hora, i per això la base és la
    predicció mitjana d'aquelles files; la suma es compleix amb qualsevol centre.
    """
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    explained = attribute_rows(model, X)
    for hour in range(24):
        rows = (X[HOUR] == hour).to_numpy()
        assert np.allclose(explained.values[rows].mean(), 0, atol=1e-9)
        assert np.isclose(
            explained.base[rows].iloc[0], explained.transformed[rows].mean()
        )


def test_moving_one_column_of_a_row_moves_only_its_own_contribution():
    """Una contribució a una altra columna sumaria igual, amb la mateixa mitjana.

    L'ordre de la matriu del model, les escalades i després les dummies, no és el de la
    matriu per hora.
    """
    X, y = per_hour_table()
    model = fit_rows(X, y, PENALTY, MIXING)
    before = attribute_rows(model, X).values
    shifted = X.copy()
    rows = (shifted[HOUR] == 13).to_numpy()
    shifted.loc[rows, "wind_fcst"] += 1000.0
    moved = attribute_rows(model, shifted).values - before
    assert (moved.loc[rows, "wind_fcst"] != 0).all()
    assert (moved.drop(columns="wind_fcst") == 0).all().all()
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


def test_each_sequence_starts_at_the_smallest_penalty_that_empties_every_hour():
    """Mal escalada, la seqüència començaria lluny del model buit, o a dins seu.

    Com una escala sense centrar o d'una sola hora; la graella semblaria igual de bona.
    """
    # Que una mica per sota el model ja no sigui buit no es pot comprovar ajustant: el
    # solucionador torna el model buit mentre la diferència li cap a la tolerància. Es
    # compara amb el primer càstig de `ElasticNetCV`, hora per hora.
    X, y = per_hour_table()
    for mixing, steps in penalties(X, y).items():
        theirs = [
            ElasticNetCV(l1_ratio=mixing, alphas=2, cv=2).fit(design, target).alphas_[0]
            for design, target in (
                _prepared(X.iloc[rows], y.iloc[rows])[1:] for rows in by_hour(X)
            )
        ]
        assert np.isclose(steps[0], max(theirs), rtol=1e-9)
        top = fit_rows(X, y, steps[0], mixing)
        assert max(np.abs(hour.model.coef_).max() for hour in top) < 1e-12


# La previsió solar a les 20 h: zero més de la meitat dels dies, milers de MW la resta,
# de manera que la seva MAD és zero tot i que es mou.
EVENING_SOLAR = np.array([0.0] * 6 + [900.0, 2500.0, 4641.0, 1200.0])


def test_the_scaler_round_trips():
    values = np.array([[0.0, 5.0], [100.0, -3.0]])
    scaler = Scaler(values, ["a", "b"])
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_each_column_gets_its_own_median_and_spread():
    # Una columna al voltant de 0 i una altra al voltant de 1000. Amb una mediana
    # compartida, la primera columna quedaria tota desplaçada cap avall.
    values = np.array([[0.0, 1000.0], [1.0, 1002.0], [2.0, 1004.0], [3.0, 1006.0]])
    scaler = Scaler(values, ["a", "b"])
    assert list(scaler.median) == [1.5, 1003.0]
    transformed = scaler.forward(values)
    assert abs(transformed[:, 0].mean()) < 1e-9
    assert abs(transformed[:, 1].mean()) < 1e-9
    assert np.allclose(scaler.back(transformed), values)


def test_a_column_without_mad_is_refused_unless_the_caller_declares_the_fallback():
    values = np.column_stack([np.arange(10.0), EVENING_SOLAR])
    with pytest.raises(Refusal):
        Scaler(values, ["a", "b"])
    scaler = Scaler(values, ["a", "b"], std_where_flat=True)
    assert scaler.scale[1] == np.std(EVENING_SOLAR)
    assert np.allclose(scaler.back(scaler.forward(values)), values)


def test_the_fallback_still_refuses_a_column_that_never_moves():
    flat = np.column_stack([np.arange(10.0), np.zeros(10)])
    with pytest.raises(Refusal, match="1 columna no té dispersió .*, b$"):
        Scaler(flat, ["a", "b"], std_where_flat=True)
