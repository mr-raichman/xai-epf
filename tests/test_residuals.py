import numpy as np
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox

from epf.run.residuals import LAGS, daily_errors, ljung_box


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


def _model(seed_days, offset, rank=0):
    """Torna un model ajustat dels dies `seed_days` amb un error `offset` per hora."""
    days = np.array(seed_days, dtype="datetime64[D]")
    actual = np.zeros(len(days) * 24)
    return {
        "rank": rank,
        "days_test": days,
        "actual": actual,
        "predicted": actual + offset,
    }


def test_each_day_comes_from_the_first_seed_that_tests_it_and_its_best_model():
    """Un dia d'una altra seed donaria un valor p plausible sobre una altra sèrie."""
    first = _model(["2020-01-01", "2020-01-02"], 1.0)
    other = _model(["2020-01-01", "2020-01-02"], 9.0, rank=1)
    second = _model(["2020-01-02", "2020-01-04"], 2.0)
    series = daily_errors([first, other, second])
    assert np.array_equal(series, [1.0, 1.0, np.nan, 2.0], equal_nan=True)
