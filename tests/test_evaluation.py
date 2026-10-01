"""Alineació del camí endavant i de l'error."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.data.calendar import HOUR_COLUMNS
from epf.evaluation import (mae, mae_per_hour, mae_per_hour_over_price,
                            walk_forward)


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
