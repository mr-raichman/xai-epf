"""Alineació dels retards per data de lliurament."""
import datetime

import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.parameters import LAGS
from epf.matrix import DAY_OF_WEEK_COLUMNS, HOUR, build, calibration, forecasts, lago, per_hour


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in days], name="day")
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_each_lag_carries_the_right_day():
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    day = X.index[10]
    for lag in LAGS:
        source = price.loc[day - pd.Timedelta(days=lag)]
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"lag{lag}_{hour}"] == source[hour]


def test_the_target_is_the_day_itself():
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    assert (y.loc[X.index[10]] == price.loc[X.index[10]]).all()


def test_the_first_complete_row_is_the_longest_lag_away():
    price = frame_of(range(30))
    X, _ = build(price, LAGS)
    assert (X.index[0] - price.index[0]).days == max(LAGS)


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


def test_the_day_of_week_is_seven_indicators_on_the_delivery_day():
    # Set indicadors i no un enter: l'enter li diu a una xarxa que diumenge és sis
    # vegades dimarts, i mentre hi era cada família el veia d'una manera diferent.
    price = frame_of(range(30))
    X, _ = build(price, LAGS)
    day = X.index[5]
    row = X.loc[day, DAY_OF_WEEK_COLUMNS]
    assert list(row) == [1.0 if number == day.dayofweek else 0.0
                         for number in range(len(DAY_OF_WEEK_COLUMNS))]
    assert X[DAY_OF_WEEK_COLUMNS].to_numpy().sum(axis=1).tolist() == [1.0] * len(X)



def test_wrong_columns_fail_loudly():
    price = frame_of(range(10)).rename(columns={"h00": "hora0"})
    with pytest.raises(ValueError):
        build(price, LAGS)


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
    price = frame_of(range(30))
    X, y = build(price, LAGS)
    rows, target = per_hour(X, y)
    assert len(rows) == len(X) * 24
    day = X.index[10]
    for hour in range(24):
        row = rows.loc[day].iloc[hour]
        assert row[HOUR] == hour
        assert row["lag1_h00"] == X.loc[day, "lag1_h00"]
        assert target.loc[day].iloc[hour] == y.loc[day, HOUR_COLUMNS[hour]]


def test_day_d_forecasts_sit_on_day_d_in_both_tables():
    # Es publiquen el dia abans i són del dia que es lliura: desplaçades un dia, cada
    # fila llegiria la previsió d'ahir, que és plausible i equivocada.
    price = frame_of(range(30))
    day_ahead = {"demand": frame_of(range(30)) * 10}
    matrices = forecasts(price, day_ahead)
    (X, _), (rows, target) = matrices.daily, matrices.hourly
    day = X.index[10]
    for hour, column in enumerate(HOUR_COLUMNS):
        assert X.loc[day, f"demand_{column}"] == day_ahead["demand"].loc[day, column]
        row = rows.loc[day].iloc[hour]
        assert row[HOUR] == hour
        assert row["demand"] == day_ahead["demand"].loc[day, column]
        assert row[f"lag1_{column}"] == price.loc[day - pd.Timedelta(days=1), column]
        assert target.loc[day].iloc[hour] == price.loc[day, column]
    assert not any(column.startswith("demand_") for column in rows.columns)


def test_the_window_counts_days_on_both_tables():
    X, y = build(frame_of(range(30)), LAGS)
    rows, target = per_hour(X, y)
    day = X.index[15]
    daily, _ = calibration(X, y, day, window=5)
    hourly, _ = calibration(rows, target, day, window=5)
    assert len(daily) == 5
    assert len(hourly) == 5 * 24
    assert list(hourly.index.unique()) == list(daily.index)


def test_each_forecast_lag_of_r1s_set_lands_on_its_own_day():
    # La previsió del dia, la del dia abans i la de fa una setmana: totes tres es
    # publiquen abans del tancament, i una desplaçada a un altre dia seria plausible.
    price = frame_of(range(30))
    day_ahead = {"demand": frame_of(range(30)) * 10, "wind": frame_of(range(30)) * 3,
                 "solar": frame_of(range(30)) + 7}
    X, _ = lago(price, day_ahead).daily
    day = X.index[-1]
    for lag in (0, 1, 7):
        source = day - pd.Timedelta(days=lag)
        prefix = "" if lag == 0 else f"lag{lag}_"
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"demand_{prefix}{hour}"] == day_ahead["demand"].loc[source, hour]
            assert X.loc[day, f"wind_solar_{prefix}{hour}"] == (
                day_ahead["wind"].loc[source, hour] + day_ahead["solar"].loc[source, hour])
    assert X.shape[1] == 96 + 2 * 3 * 24 + 7


def test_a_window_that_cannot_be_filled_is_refused():
    # El conjunt de Lago et al. (2021) comença el 2019-01-08 i la finestra de quatre anys
    # no s'omple fins al final del 2022: més curta, passaria sense dir res.
    X, y = build(frame_of(range(30)), LAGS)
    day = X.index[-1]
    available = len(X.index[X.index < day])
    assert len(calibration(X, y, day, window=available)[0]) == available
    with pytest.raises(ValueError, match="la finestra en demana"):
        calibration(X, y, day, window=available + 1)
