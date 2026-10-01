"""Alineació del camí endavant i de l'error."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.data.calendar import HOUR_COLUMNS
from epf.evaluation import (curve_correlations, error_dispersion, mae, mae_per_hour,
                            mae_per_hour_over_price, rmae, rmae_per_hour, walk_forward)
from epf.models.naive import weekly_frame


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in range(days)], name="day")
    rows = [[d * 100 + hour + offset for hour in range(24)] for d in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def perfect(X, y, day):
    """Un model que encerta, per comprovar l'alineació i no el model."""
    return y.loc[day].to_numpy()


def test_every_day_of_the_period_is_predicted_and_kept_in_order():
    X = y = frame_of(30)
    out = walk_forward(perfect, X, y, "2024-01-10", "2024-01-20")
    assert list(out.index) == list(pd.date_range("2024-01-10", "2024-01-20", freq="D"))
    assert list(out.columns) == HOUR_COLUMNS


def test_each_row_is_the_day_it_is_labelled_with():
    X = y = frame_of(30)
    out = walk_forward(perfect, X, y, "2024-01-10", "2024-01-20")
    assert (out == y.loc[out.index]).to_numpy().all()
    assert mae(out, y) == 0.0


def test_taking_one_day_in_every_n_keeps_the_first():
    X = y = frame_of(30)
    out = walk_forward(perfect, X, y, "2024-01-10", "2024-01-20", every=5)
    assert list(out.index) == [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-15"),
                              pd.Timestamp("2024-01-20")]


def test_the_error_is_measured_only_on_the_days_both_tables_hold():
    X = y = frame_of(30)
    out = walk_forward(perfect, X, y, "2024-01-10", "2024-01-12")
    shifted = frame_of(30, offset=3.0)
    assert mae(out, shifted) == pytest.approx(3.0)
    assert np.allclose(mae_per_hour(out, shifted).to_numpy(), 3.0)


def test_an_empty_period_fails_loudly():
    X = y = frame_of(30)
    with pytest.raises(ValueError):
        walk_forward(perfect, X, y, "2025-01-01", "2025-02-01")


def test_tables_with_no_day_in_common_fail_loudly():
    X = y = frame_of(30)
    out = walk_forward(perfect, X, y, "2024-01-10", "2024-01-12")
    with pytest.raises(ValueError):
        mae(out, frame_of(5, start=datetime.date(2030, 1, 1)))


def test_a_day_repeated_in_the_index_is_predicted_once():
    # La taula per hora té cada dia 24 vegades i només s'ha de predir una.
    y = frame_of(30)
    X = y.loc[y.index.repeat(24)]
    out = walk_forward(lambda features, target, day: target.loc[day].to_numpy(),
                       X, y, "2024-01-10", "2024-01-12")
    assert list(out.index) == list(pd.date_range("2024-01-10", "2024-01-12", freq="D"))


def test_the_normalised_error_divides_by_that_hour_s_mean_price():
    actual = frame_of(30)
    predicted = frame_of(30, offset=3.0).loc["2024-01-10":"2024-01-12"]
    level = actual.loc[predicted.index].mean()
    assert np.allclose(mae_per_hour_over_price(predicted, actual).to_numpy(),
                       3.0 / level.to_numpy())
    # El preu mitjà surt només dels dies compartits, no de tota la sèrie.
    assert not np.allclose(mae_per_hour_over_price(predicted, actual).to_numpy(),
                           3.0 / actual.mean().to_numpy())


def test_a_mean_price_at_or_below_zero_fails_loudly():
    with pytest.raises(ValueError):
        mae_per_hour_over_price(frame_of(3), frame_of(3) * 0.0)


def test_the_weekly_naive_is_the_same_hour_seven_days_earlier():
    prices = frame_of(30)
    naive = weekly_frame(prices)
    # Els set primers dies no tenen el seu dia de fa una setmana i no hi són.
    assert list(naive.index) == list(prices.index[7:])
    day = naive.index[5]
    assert (naive.loc[day] == prices.loc[day - pd.Timedelta(days=7)]).all()


def test_the_naive_forecast_scores_exactly_one_against_itself():
    prices = frame_of(60)
    naive = weekly_frame(prices)
    assert rmae(naive, prices, naive) == pytest.approx(1.0)
    assert np.allclose(rmae_per_hour(naive, prices, naive).to_numpy(), 1.0)


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
