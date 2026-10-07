"""Alineació de l'error, i les mesures que podrien equivocar-se en silenci.

El camí endavant que produeix les prediccions és a `tests/test_walk.py`: aquí les
taules es construeixen a mà, de manera que el que falla quan una prova falla és la
mesura.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.evaluation import (curve_correlations, error_dispersion,
                            extrema_price_deviations, mae, mae_per_hour, metrics,
                            refuse_missing, rmae, rmae_per_hour)
from epf.models.naive.naive import weekly_frame


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[d * 100 + hour + offset for hour in range(24)] for d in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def three_days(offset=0.0):
    """Tres dies d'una taula de trenta, que és el que un pronòstic curt cobreix."""
    return frame_of(30, offset=offset).loc["2024-01-10":"2024-01-12"]


def test_the_error_is_measured_only_on_the_days_both_tables_hold():
    assert mae(three_days(), frame_of(30, offset=3.0)) == pytest.approx(3.0)
    assert np.allclose(
        mae_per_hour(three_days(), frame_of(30, offset=3.0)).to_numpy(), 3.0)


def test_tables_with_no_day_in_common_fail_loudly():
    with pytest.raises(ValueError):
        mae(three_days(), frame_of(5, start=datetime.date(2030, 1, 1)))


def test_a_missing_value_fails_instead_of_being_skipped():
    # `mae` tornaria NaN, que es veu, però `mae_per_hour` el saltaria i informaria
    # l'hora com si fos neta: aquest és el cas que la guarda existeix per agafar.
    holed = frame_of(30, offset=3.0)
    holed.iloc[4, 0] = np.nan
    assert mae_per_hour(holed, frame_of(30))["h00"] == 3.0
    with pytest.raises(ValueError, match="h00"):
        refuse_missing(holed, "el pronòstic")


def test_the_metrics_refuse_a_forecast_with_a_hole():
    actual = frame_of(30)
    holed = frame_of(30, offset=3.0)
    holed.iloc[4, 0] = np.nan
    with pytest.raises(ValueError):
        metrics({"model": holed}, actual, weekly_frame(actual))


def test_the_weekly_naive_is_the_same_hour_seven_days_earlier():
    price = frame_of(30)
    naive = weekly_frame(price)
    # Els set primers dies no tenen el seu dia de fa una setmana i no hi són.
    assert list(naive.index) == list(price.index[7:])
    day = naive.index[5]
    assert (naive.loc[day] == price.loc[day - pd.Timedelta(days=7)]).all()


def test_the_naive_forecast_scores_exactly_one_against_itself():
    price = frame_of(60)
    naive = weekly_frame(price)
    assert rmae(naive, price, naive) == pytest.approx(1.0)
    assert np.allclose(rmae_per_hour(naive, price, naive).to_numpy(), 1.0)


def test_the_curve_correlation_is_on_ranks_and_not_on_levels():
    actual = frame_of(3)
    # Spearman mira l'ordre de les hores, no els valors: doblar-los no el mou.
    assert np.allclose(curve_correlations(actual * 2.0, actual).to_numpy(), 1.0)
    # Invertir l'ordre de les hores del dia el porta a -1.
    reversed_hours = actual.iloc[:, ::-1]
    reversed_hours.columns = actual.columns
    assert np.allclose(curve_correlations(reversed_hours, actual).to_numpy(), -1.0)


def test_a_day_with_no_spread_has_no_curve_correlation():
    actual = frame_of(3)
    flat = actual.copy()
    flat.iloc[1] = 50.0
    correlations = curve_correlations(flat, actual)
    assert np.isnan(correlations.iloc[1])
    assert correlations.notna().sum() == 2


def test_the_error_dispersion_needs_more_days_than_hours():
    actual = frame_of(23)
    with pytest.raises(ValueError):
        error_dispersion(frame_of(23, offset=1.0), actual)


def test_scaling_every_error_moves_the_dispersion_by_a_known_amount():
    # La matriu té 24 files: multiplicar tots els errors per k multiplica el
    # determinant per k^48, i el seu logaritme puja 48*log(k).
    actual = frame_of(60)
    rng = np.random.default_rng(0)
    noisy = actual + rng.normal(0.0, 5.0, size=actual.shape)
    doubled = actual + (noisy - actual) * 2.0
    moved = error_dispersion(doubled, actual) - error_dispersion(noisy, actual)
    assert moved == pytest.approx(48 * np.log(2.0))


def test_hitting_both_extreme_hours_loses_no_spread():
    actual = frame_of(5)
    assert np.allclose(extrema_price_deviations(actual, actual).to_numpy(), 0.0)


def test_predicting_the_day_backwards_loses_twice_the_spread():
    # El pronòstic col·loca el mínim on hi ha el màxim i a l'inrevés, de manera que es
    # perd la forquilla dues vegades: una per cada extrem.
    actual = frame_of(5)
    backwards = actual.iloc[:, ::-1]
    backwards.columns = actual.columns
    spread = actual.max(axis="columns") - actual.min(axis="columns")
    assert np.allclose(extrema_price_deviations(backwards, actual).to_numpy(),
                       2 * spread.to_numpy())


def test_the_deviation_reads_the_real_price_at_the_predicted_hour():
    # Un pronòstic pla excepte una punta a una hora qualsevol: el que compta no és el
    # valor que prediu, sinó el preu real de l'hora que tria.
    actual = frame_of(1)
    guess = actual.copy()
    guess.iloc[0] = 0.0
    guess.iloc[0, 3] = 1.0
    guess.iloc[0, 9] = -1.0
    lost = extrema_price_deviations(guess, actual).iloc[0]
    row = actual.iloc[0]
    assert lost == pytest.approx(abs(row.min() - row.iloc[9]) + abs(row.max() - row.iloc[3]))
