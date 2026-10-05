"""Alineació dels retards per data de lliurament."""
import datetime

import pandas as pd
import pytest

from epf.data.calendar import HOUR_COLUMNS
from epf.config.experiments import LAGS
from epf.features.matrix import DAY_OF_WEEK, HOUR, build, calibration, per_hour


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in days], name="day")
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_each_lag_carries_the_right_day():
    prices = frame_of(range(30))
    X, y = build(prices, LAGS)
    day = X.index[10]
    for lag in LAGS:
        source = prices.loc[day - pd.Timedelta(days=lag)]
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"lag{lag}_{hour}"] == source[hour]


def test_the_target_is_the_day_itself():
    prices = frame_of(range(30))
    X, y = build(prices, LAGS)
    assert (y.loc[X.index[10]] == prices.loc[X.index[10]]).all()


def test_the_first_complete_row_is_the_longest_lag_away():
    prices = frame_of(range(30))
    X, _ = build(prices, LAGS)
    assert (X.index[0] - prices.index[0]).days == max(LAGS)


def test_a_gap_removes_the_rows_that_need_it_instead_of_shifting_them():
    complete = build(frame_of(range(30)), LAGS)[0]
    with_gap = build(frame_of([d for d in range(30) if d != 15]), LAGS)[0]

    gap = pd.Timestamp("2020-01-16")
    missing = set(complete.index) - set(with_gap.index)
    assert missing == {gap} | {gap + pd.Timedelta(days=lag) for lag in LAGS}

    # Les files restants segueixen aparellades amb el seu dia.
    day = with_gap.index[-1]
    offset = (day - pd.Timedelta(days=1) - pd.Timestamp("2020-01-01")).days
    assert with_gap.loc[day, "lag1_h00"] == offset * 100


def test_day_of_week_is_the_delivery_day():
    prices = frame_of(range(30))
    X, _ = build(prices, LAGS)
    day = X.index[5]
    assert X.loc[day, DAY_OF_WEEK] == day.dayofweek



def test_wrong_columns_fail_loudly():
    prices = frame_of(range(10)).rename(columns={"h00": "hora0"})
    with pytest.raises(ValueError):
        build(prices, LAGS)


def test_the_calibration_slice_never_reaches_the_day_itself():
    X, y = build(frame_of(range(30)), LAGS)
    day = X.index[10]
    features, target = calibration(X, y, day)
    assert features.index.max() < day
    assert target.index.equals(features.index)


def test_the_window_takes_the_last_days_before_it():
    X, y = build(frame_of(range(30)), LAGS)
    day = X.index[15]
    features, _ = calibration(X, y, day, window=5)
    assert list(features.index) == list(X.index[10:15])


def test_calibrating_with_no_earlier_day_fails_loudly():
    X, y = build(frame_of(range(30)), LAGS)
    with pytest.raises(ValueError):
        calibration(X, y, X.index[0])


def test_the_per_hour_row_carries_its_day_and_its_own_hour():
    prices = frame_of(range(30))
    X, y = build(prices, LAGS)
    rows, target = per_hour(X, y)
    assert len(rows) == len(X) * 24
    day = X.index[10]
    for hour in range(24):
        row = rows.loc[day].iloc[hour]
        assert row[HOUR] == hour
        assert row["lag1_h00"] == X.loc[day, "lag1_h00"]
        assert target.loc[day].iloc[hour] == y.loc[day, HOUR_COLUMNS[hour]]


def test_the_window_counts_days_on_both_tables():
    X, y = build(frame_of(range(30)), LAGS)
    rows, target = per_hour(X, y)
    day = X.index[15]
    daily, _ = calibration(X, y, day, window=5)
    hourly, _ = calibration(rows, target, day, window=5)
    assert len(daily) == 5
    assert len(hourly) == 5 * 24
    assert list(hourly.index.unique()) == list(daily.index)
