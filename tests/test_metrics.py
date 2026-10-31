import warnings

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from epf import Refusal
from epf.contract.metrics import (
    LAGS,
    _extremes,
    corrected_t_test,
    daily_errors,
    diebold_mariano,
    ljung_box,
    one_seed,
    part_i_main,
    part_i_secondary,
    part_i_tests,
    scored,
)


def test_a_score_refuses_a_prediction_that_is_not_a_number_and_fails_a_missing_price():
    with pytest.raises(Refusal, match="la predicció té 2 valors no finits"):
        scored([1.0, np.nan, np.inf], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="el preu real té 1 valor absent") as raised:
        scored([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
    assert not isinstance(raised.value, Refusal)


def _values(rows, measure):
    """Torna el valor de cada fila d'una mesura, per família, seed i lloc."""
    return {
        (row["family"], row.get("seed"), row.get("rank")): row["value"]
        for row in rows
        if row["measure"] == measure
    }


def test_the_main_metrics_divide_by_the_copy_and_summarise_each_seed_s_best():
    # L'arbre a la seed 0, MAE 10 i 12, i a la 1, 9; la còpia, 20 i 18. rMAE 10 / 20 =
    # 0.5, 12 / 20 = 0.6 i 9 / 18 = 0.5. Sobre les millors, el MAE de l'arbre té mitjana
    # 9.5 i desviació mostral √((0.5² + 0.5²) / 1) = √0.5, i l'rMAE, 0.5 i 0.
    scores = {
        "lightgbm": {
            (0, 0): {"r2": 0.5, "mae": 10.0},
            (0, 1): {"r2": 0.25, "mae": 12.0},
            (1, 0): {"r2": 0.75, "mae": 9.0},
        },
        "naive_previous_week": {
            (0, 0): {"r2": 0.0, "mae": 20.0},
            (1, 0): {"r2": 0.1, "mae": 18.0},
        },
    }
    rows = part_i_main(scores, "naive_previous_week", ["lightgbm"])
    assert _values(rows, "rmae") == {
        ("lightgbm", 0, 0): 0.5,
        ("lightgbm", 0, 1): 0.6,
        ("lightgbm", 1, 0): 0.5,
        ("naive_previous_week", 0, 0): 1.0,
        ("naive_previous_week", 1, 0): 1.0,
    }
    assert _values(rows, "mae_mean") == {("lightgbm", None, 0): 9.5}
    assert _values(rows, "mae_std") == {("lightgbm", None, 0): pytest.approx(0.5**0.5)}
    assert _values(rows, "rmae_std") == {("lightgbm", None, 0): 0.0}
    assert {row["other_family"] for row in rows if "rmae" in row["measure"]} == {
        "naive_previous_week"
    }


def test_a_seed_without_the_copy_has_no_rmae_nor_its_family_a_summary_of_it():
    scores = {
        "lightgbm": {(0, 0): {"r2": 0.5, "mae": 10.0}, (1, 0): {"r2": 0.7, "mae": 9.0}},
        "naive_previous_week": {(0, 0): {"r2": 0.0, "mae": 20.0}},
    }
    rows = part_i_main(scores, "naive_previous_week", ["lightgbm"])
    assert _values(rows, "rmae") == {
        ("lightgbm", 0, 0): 0.5,
        ("naive_previous_week", 0, 0): 1.0,
    }
    assert not _values(rows, "rmae_mean") and _values(rows, "mae_mean")


def test_one_seed_s_spread_is_a_blank_value_with_no_warning():
    scores = {"floor": {(0, 0): {"r2": 0.5, "mae": 10.0}}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = part_i_main(scores, None, ["floor"])
    assert np.isnan(_values(rows, "mae_std")[("floor", None, 0)])
    assert not _values(rows, "rmae")


def test_without_gaps_the_statistic_is_the_usual_ljung_box():
    """Un estadístic mal calculat donaria un valor p plausible."""
    # Una AR(1) de coeficient 0.3: una implementació que no és la del projecte ha de
    # donar el mateix estadístic i el mateix valor p.
    rng = np.random.default_rng(0)
    series = np.zeros(500)
    for t in range(1, len(series)):
        series[t] = 0.3 * series[t - 1] + rng.normal()
    statistic, value, _ = ljung_box(series)
    expected = acorr_ljungbox(series, lags=[LAGS])
    assert statistic == pytest.approx(expected["lb_stat"].iloc[0])
    assert value == pytest.approx(expected["lb_pvalue"].iloc[0])


def test_a_missing_day_is_left_out_of_its_pairs_as_r_does():
    # Feta a mà amb l'`acf0` de R: la mitjana dels presents és 3, la covariància a 0 és
    # 10/4, la del retard 1 són dos parells que sumen 4 dividits per 2 + 1, r = 8/15, i
    # amb n = 4, Q = 4 · 6 · r^2 / 3.
    statistic, _, correlations = ljung_box(
        np.array([1.0, 2.0, np.nan, 4.0, 5.0]), lags=1
    )
    assert correlations[0] == pytest.approx(8 / 15)
    assert statistic == pytest.approx(4 * 6 * (8 / 15) ** 2 / 3)


def test_each_day_comes_from_the_first_seed_that_tests_it():
    """Un dia d'una altra seed donaria un valor p plausible sobre una altra sèrie."""
    first = (np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"), 1.0)
    second = (np.array(["2020-01-02", "2020-01-04"], dtype="datetime64[D]"), 2.0)
    series = daily_errors(
        [(days, np.full((len(days), 24), error)) for days, error in (first, second)]
    )
    assert np.array_equal(series, [1.0, 1.0, np.nan, 2.0], equal_nan=True)


def _day(mean):
    """Torna els 24 preus d'un dia de mitjana `mean`, que pugen 0.1 cada hora."""
    return mean + (np.arange(24) - 11.5) * 0.1


def test_the_errors_metrics_by_hand():
    # Dues setmanes de dimarts a dilluns: la primera a 10 de mitjana i errors d'1, la
    # segona a 20 i errors de -4. WMAE (1 / 10 + 4 / 20) / 2 = 0.15; WAPE 5 / 30; error
    # mitjà -1.5; MdAE 2.5, a mig camí d'1 i de 4; MSE (1 + 16) / 2 = 8.5.
    days = np.arange("2019-01-01", "2019-01-15", dtype="datetime64[D]")
    actual = np.array([_day(10 if day < 7 else 20) for day in range(14)])
    predicted = actual - np.where(np.arange(14) < 7, 1.0, -4.0)[:, None]
    found = one_seed(days, actual, predicted)
    assert found["wmae"] == pytest.approx(0.15)
    assert found["wape"] == pytest.approx(5 / 30)
    assert found["mean_error"] == pytest.approx(-1.5)
    assert found["mdae"] == pytest.approx(2.5)
    assert found["mse"] == pytest.approx(8.5)
    assert found["rmse"] == pytest.approx(8.5**0.5)


def test_the_correlations_and_cov_e_match_other_implementations():
    # Preus arrodonits, que empaten com els de les hores a zero.
    rng = np.random.default_rng(0)
    actual = np.round(rng.normal(50, 20, (30, 24)), -1)
    predicted = np.round(actual + rng.normal(0, 8, (30, 24)), -1)
    days = np.arange("2019-01-01", "2019-01-31", dtype="datetime64[D]")
    found = one_seed(days, actual, predicted)
    error = actual - predicted
    assert found["corr"] == pytest.approx(
        stats.pearsonr(actual.ravel(), predicted.ravel()).statistic
    )
    assert found["corr_f"] == pytest.approx(
        np.mean([stats.spearmanr(a, f).statistic for a, f in zip(actual, predicted)])
    )
    eigenvalues = np.linalg.eigvalsh(error.T @ error / len(error))
    assert found["cov_e"] == pytest.approx(np.sum(np.log(eigenvalues)))


def test_a_forecast_on_any_of_the_tied_true_hours_hits_the_extreme():
    # Dos dies a 50, a zero de les 10 a les 15 i a 100 a les 20. El primer previst té
    # el mínim a les 14, una de les zero, i el màxim a les 19: 0 + 1 hores, 0 + 50
    # EUR/MWh. El segon empata el mínim a les 3 i a les 12, i pren les 3, a 7 hores de
    # les 10 i a 50 EUR/MWh, i encerta el màxim. MHD (1 + 7) / 2 = 4, MPD 50.
    actual = np.full((2, 24), 50.0)
    actual[:, 10:16] = 0.0
    actual[:, 20] = 100.0
    predicted = np.full((2, 24), 40.0)
    predicted[0, 14], predicted[0, 19] = 1.0, 90.0
    predicted[1, [3, 12]], predicted[1, 20] = 2.0, 95.0
    distance, price = _extremes(actual, predicted)
    assert np.array_equal(distance, [1, 7]) and np.array_equal(price, [50, 50])


def _fit(days, errors, **arrays):
    """Torna la millor configuració d'una seed: dies de mitjana 10, 20, ... i errors."""
    days = np.array(days, dtype="datetime64[D]")
    actual = np.array([_day(10 * (number + 1)) for number in range(len(days))])
    return {"days": days, "actual": actual, "predicted": actual - errors, **arrays}


def _secondary(rows, measure, family=None, **keys):
    """Torna el valor de les files d'una mesura i una família que tenen les claus."""
    return [
        row["value"]
        for row in rows
        if row["measure"] == measure
        and row["family"] == family
        and all(row.get(key) == value for key, value in keys.items())
    ]


def test_the_secondary_metrics_against_the_copies_by_hand():
    """La sonda de datació no entra entre les millors, i pot quedar-ne per sota."""
    # Seed 0 al 2019 i seed 1 al 2020, dos dies cada una. Errors constants: l'arbre 1
    # i 3, la sonda 0.5, la còpia de la setmana -2, la del dia 4. rMAE sobre la del dia
    # 1 / 4; MASE 1 / 4 i RMSSE √(1 / 25), sobre l'entrenament de la còpia; la
    # desviació de la millor, entre l'arbre i les còpies, 0 per a l'arbre i -0.5 per a
    # la sonda. Per any, l'arbre fa 1 / 2 el 2019 i 3 / 2 el 2020, cada any d'una seed.
    # A cada hora, els dos dies valen a i a + 10: l'R² sobre la mitjana de l'hora és
    # 1 - 2 / 50 a la seed 0, i amb la de la seed sencera seria un altre.
    first, second = ["2019-01-01", "2019-01-02"], ["2020-01-07", "2020-01-08"]
    best = {
        "lightgbm": {0: _fit(first, 1.0), 1: _fit(second, 3.0)},
        "date_probe": {0: _fit(first, 0.5), 1: _fit(second, 0.5)},
        "naive_previous_week": {
            seed: _fit(days, -2.0, training_mae=4.0, training_mse=25.0)
            for seed, days in ((0, first), (1, second))
        },
        "naive_previous_day": {0: _fit(first, 4.0), 1: _fit(second, 4.0)},
    }
    roles = {
        "lightgbm": {"interpreted": True},
        "date_probe": {},
        "naive_previous_week": {"benchmark": True, "denominator": "main"},
        "naive_previous_day": {"benchmark": True, "denominator": "secondary"},
    }
    rows = part_i_secondary(best, roles)
    assert _secondary(
        rows, "rmae", "lightgbm", seed=0, other_family="naive_previous_day"
    ) == [0.25]
    assert _secondary(rows, "mase", "lightgbm", seed=0) == [0.25]
    assert _secondary(rows, "rmsse", "lightgbm", seed=0) == [pytest.approx(0.2)]
    assert _secondary(rows, "deviation_from_best", "lightgbm", seed=0) == [0.0]
    assert _secondary(rows, "deviation_from_best", "date_probe", seed=0) == [-0.5]
    assert _secondary(rows, "r2", "lightgbm", seed=0, hour=5) == [pytest.approx(0.96)]
    assert _secondary(rows, "rmae_year", "lightgbm", seed=0, subject="2019") == [0.5]
    assert np.isnan(
        _secondary(rows, "rmae_year", "lightgbm", seed=0, subject="2020")[0]
    )
    means = [
        row["value"]
        for row in rows
        if row["measure"] == "rmae_year"
        and row["family"] == "lightgbm"
        and row.get("seed") is None
    ]
    assert means == [0.5, 1.5]
    prices = np.concatenate([fit["actual"] for fit in best["lightgbm"].values()])
    assert _secondary(rows, "price_decile_top") == list(
        np.quantile(prices, np.linspace(0, 1, 11))[1:]
    )


def test_the_dm_tests_by_hand():
    # Errors de la primera 2, 4 i 6 a totes les hores, i de la segona 1: diferències
    # 1, 3 i 5, mitjana 3, variància sense corregir 8 / 3, i l'estadístic
    # 3 / √(8 / 3 / 3) = 9 / (2√2), al dia i a cada hora.
    first = np.repeat([[2.0], [-4.0], [6.0]], 24, axis=1)
    second = np.ones((3, 24))
    day, hours = diebold_mariano(first, second)
    expected = 1 - stats.norm.cdf(9 / (2 * 2**0.5))
    assert day == pytest.approx(expected) and hours == pytest.approx([expected] * 24)
    assert diebold_mariano(second, first)[0] == pytest.approx(1 - expected)


def test_the_corrected_t_test_by_hand():
    # Quatre seeds, diferències 1, 2, 3 i 6: mitjana 3, variància mostral 14 / 3, i la
    # proporció de setmanes de la partició, 80 de prova sobre 193 + 129.
    ratio = 80 / 322
    scale = ((1 / 4 + ratio) * 14 / 3) ** 0.5
    found = corrected_t_test(np.array([1.0, 2.0, 3.0, 6.0]), ratio, 0.5)
    assert found["t_statistic"] == pytest.approx(3 / scale)
    assert found["t_test"] == pytest.approx(2 * stats.t.sf(3 / scale, 3))
    assert found["t_left"] == pytest.approx(stats.t.cdf(-0.5, 3, loc=3, scale=scale))
    assert found["t_right"] == pytest.approx(stats.t.sf(0.5, 3, loc=3, scale=scale))
    assert sum(
        found[name] for name in ("t_left", "t_rope", "t_right")
    ) == pytest.approx(1)
    assert all(
        np.isnan(value) for value in corrected_t_test(np.ones(4), ratio, 0.5).values()
    )


def _parted(days, errors, weeks):
    """Torna una seed de `part_i_tests`: dies, errors, i setmanes de cada part."""
    start = np.datetime64("2019-01-01")
    return _fit(days, errors) | {
        "days_training": start + 7 * np.arange(weeks[0]),
        "days_validation": start + 7 * np.arange(weeks[0], weeks[0] + weeks[1]),
    }


def test_each_pair_is_tested_both_ways_by_dm_once_by_the_t_test():
    first, second = ["2019-01-01", "2019-01-02"], ["2020-01-07", "2020-01-08"]
    best = {
        name: {0: _parted(first, error, (2, 1)), 1: _parted(second, error + 1, (2, 1))}
        for name, error in (
            ("lightgbm", np.array([[1.0], [1.5]])),
            ("naive_previous_week", np.array([[2.0], [3.0]])),
        )
    }
    rows = part_i_tests(best)
    assert len(_secondary(rows, "dm_day", "lightgbm", seed=0)) == 1
    assert len(_secondary(rows, "dm_day", "naive_previous_week", seed=0)) == 1
    assert len(_secondary(rows, "dm_hour", "lightgbm", seed=1)) == 24
    assert len(_secondary(rows, "error_correlation", "lightgbm")) == 2
    assert not _secondary(rows, "error_correlation", "naive_previous_week")
    assert [row.get("seed") for row in rows if row["measure"] == "t_test"] == [None]
    best["lightgbm"][1]["days_validation"] = best["lightgbm"][1]["days_validation"][:0]
    with pytest.raises(ValueError, match="proporcions de setmanes"):
        part_i_tests(best)
