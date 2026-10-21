import numpy as np
import pandas as pd
import pytest
from scipy import stats

from epf import Refusal
from epf.contract.metrics import SIGNIFICANCE, kupiec, scored
from epf.parameters import HOUR_COLUMNS


def _hits_per_hour(inside, days):
    """Torna uns preus de `days` dies i els extrems d'un interval de [0, 1] per preu.

    A l'hora `h`, `inside[h]` preus són a dins, a 0.5, i la resta a fora, a 2.
    """
    price = np.full((days, len(inside)), 2.0)
    for hour, count in enumerate(inside):
        price[:count, hour] = 0.5
    bounds = np.stack([np.zeros_like(price), np.ones_like(price)], axis=-1)
    return pd.DataFrame(price, columns=HOUR_COLUMNS[: len(inside)]), bounds


def test_kupiecs_statistic_is_the_binomial_likelihood_ratio_of_the_coverage():
    # La primera hora és a la cobertura nominal, i no rebutja res.
    days, coverage = 100, 0.9
    inside = [90, 80, 97, 60]
    actual, bounds = _hits_per_hour(inside, days)
    found = kupiec(actual, bounds, coverage)
    ratio = 2 * (
        stats.binom.logpmf(inside, days, np.array(inside) / days)
        - stats.binom.logpmf(inside, days, coverage)
    )
    assert found[0] == pytest.approx(1.0)
    assert found == pytest.approx(stats.chi2.sf(ratio, 1))


def test_an_hour_that_never_misses_gets_a_value_and_not_a_nan():
    # En deu dies, totes dins d'un interval del 90% és una cobertura que passa.
    actual, bounds = _hits_per_hour([10], 10)
    found = kupiec(actual, bounds, 0.9)
    assert np.isfinite(found).all()
    assert found[0] > SIGNIFICANCE


def test_a_score_refuses_a_prediction_that_is_not_a_number_and_fails_a_missing_price():
    with pytest.raises(Refusal, match="la predicció té 2 valors no finits"):
        scored([1.0, np.nan, np.inf], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="el preu real té 1 valor absent"):
        scored([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])


def test_a_price_exactly_on_its_bound_counts_inside_whatever_the_rounding():
    # El dia similar del 2022-12-25 a les 23:00: la còpia, 89.00, més l'error del 5% de
    # l'hora, el del 2024-09-14, 80.98 - 123.86, fa l'extrem de 46.12, el preu mateix, i
    # la suma en coma flotant en queda a 46.120000000000005; amb l'error desat en
    # float32, a 46.11999893.
    error = 80.98 - 123.86
    price = pd.DataFrame({HOUR_COLUMNS[0]: [46.12] * 10})
    inside = np.tile([[[40.0, 50.0]]], (10, 1, 1))
    for lower in (89.0 + error, 89.0 + float(np.float32(error))):
        tied = inside.copy()
        tied[0, 0, 0] = lower
        assert kupiec(price, tied, 0.9) == pytest.approx(kupiec(price, inside, 0.9))
