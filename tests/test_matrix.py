"""Alineació dels retards per data de lliurament."""
import datetime

import pandas as pd
import pytest

from epf.data.calendar import HOUR_COLUMNS
from epf.features.matrix import DAY_OF_WEEK, PRICE_LAGS, build, feature_names


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in days], name="day")
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_each_lag_carries_the_right_day():
    prices = frame_of(range(30))
    X, y = build(prices)
    day = X.index[10]
    for lag in PRICE_LAGS:
        source = prices.loc[day - pd.Timedelta(days=lag)]
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"lag{lag}_{hour}"] == source[hour]


def test_the_target_is_the_day_itself():
    prices = frame_of(range(30))
    X, y = build(prices)
    assert (y.loc[X.index[10]] == prices.loc[X.index[10]]).all()


def test_the_first_complete_row_is_the_longest_lag_away():
    prices = frame_of(range(30))
    X, _ = build(prices)
    assert (X.index[0] - prices.index[0]).days == max(PRICE_LAGS)


def test_a_gap_removes_the_rows_that_need_it_instead_of_shifting_them():
    complete = build(frame_of(range(30)))[0]
    with_gap = build(frame_of([d for d in range(30) if d != 15]))[0]

    gap = pd.Timestamp("2020-01-16")
    missing = set(complete.index) - set(with_gap.index)
    assert missing == {gap} | {gap + pd.Timedelta(days=lag) for lag in PRICE_LAGS}

    # Les files restants segueixen aparellades amb el seu dia.
    day = with_gap.index[-1]
    offset = (day - pd.Timedelta(days=1) - pd.Timestamp("2020-01-01")).days
    assert with_gap.loc[day, "lag1_h00"] == offset * 100


def test_day_of_week_is_the_delivery_day():
    prices = frame_of(range(30))
    X, _ = build(prices)
    day = X.index[5]
    assert X.loc[day, DAY_OF_WEEK] == day.dayofweek


def test_columns_are_in_the_declared_order():
    X, _ = build(frame_of(range(30)))
    assert list(X.columns) == feature_names()


def test_wrong_columns_fail_loudly():
    prices = frame_of(range(10)).rename(columns={"h00": "hora0"})
    with pytest.raises(ValueError):
        build(prices)
