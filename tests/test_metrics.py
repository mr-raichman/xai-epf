import datetime

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from epf.contract.metrics import SIGNIFICANCE, kupiec, refuse_missing
from epf.parameters import HOUR_COLUMNS


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    """Torna uns preus sintètics on cada valor és dia * 100 + hora + `offset`."""
    index = pd.DatetimeIndex(
        [start + datetime.timedelta(days=d) for d in range(days)], name="day"
    )
    rows = [[d * 100 + hour + offset for hour in range(24)] for d in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_a_missing_value_fails_naming_where_it_is():
    holed = frame_of(30, offset=3.0)
    holed.iloc[4, 0] = np.nan
    with pytest.raises(ValueError, match="1 valor absent, a .* 00:00"):
        refuse_missing(holed, "el preu real")


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
