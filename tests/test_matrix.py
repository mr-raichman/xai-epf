"""Alineació dels retards per data de lliurament."""
import datetime

import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS, HOUR, build, per_hour, schedule
from epf.parameters import PREVIOUS_DAY


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex([start + datetime.timedelta(days=d) for d in days], name="day")
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def test_each_lag_carries_the_right_day():
    price = frame_of(range(30))
    X, y = build(price, PREVIOUS_DAY)
    day = X.index[10]
    for lag in PREVIOUS_DAY:
        source = price.loc[day - pd.Timedelta(days=lag)]
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"lag{lag}_{hour}"] == source[hour]


def test_the_target_is_the_day_itself():
    price = frame_of(range(30))
    X, y = build(price, PREVIOUS_DAY)
    assert (y.loc[X.index[10]] == price.loc[X.index[10]]).all()


def test_the_first_complete_row_is_the_longest_lag_away():
    price = frame_of(range(30))
    X, _ = build(price, PREVIOUS_DAY)
    assert (X.index[0] - price.index[0]).days == max(PREVIOUS_DAY)


def test_a_gap_removes_the_rows_that_need_it_instead_of_shifting_them():
    complete = build(frame_of(range(30)), PREVIOUS_DAY)[0]
    with_gap = build(frame_of([d for d in range(30) if d != 15]), PREVIOUS_DAY)[0]

    gap = pd.Timestamp("2020-01-16")
    missing = set(complete.index) - set(with_gap.index)
    assert missing == {gap} | {gap + pd.Timedelta(days=lag) for lag in PREVIOUS_DAY}

    # Les files restants segueixen aparellades amb el seu dia.
    day = with_gap.index[-1]
    offset = (day - pd.Timedelta(days=1) - pd.Timestamp("2020-01-01")).days
    assert with_gap.loc[day, "lag1_h00"] == offset * 100


def test_the_day_of_week_is_seven_indicators_on_the_delivery_day():
    # Set indicadors i no un enter: l'enter li diu a una xarxa que diumenge és sis
    # vegades dimarts, i mentre hi era cada família el veia d'una manera diferent.
    price = frame_of(range(30))
    X, _ = build(price, PREVIOUS_DAY)
    day = X.index[5]
    row = X.loc[day, DAY_OF_WEEK_COLUMNS]
    assert list(row) == [1.0 if number == day.dayofweek else 0.0
                         for number in range(len(DAY_OF_WEEK_COLUMNS))]
    assert X[DAY_OF_WEEK_COLUMNS].to_numpy().sum(axis=1).tolist() == [1.0] * len(X)



def test_wrong_columns_fail_loudly():
    price = frame_of(range(10)).rename(columns={"h00": "hora0"})
    with pytest.raises(ValueError):
        build(price, PREVIOUS_DAY)


def test_the_per_hour_row_carries_its_day_and_its_own_hour():
    price = frame_of(range(30))
    X, y = build(price, PREVIOUS_DAY)
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
    matrices = schedule(price, day_ahead, {})
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


def test_d_minus_one_schedule_sits_on_day_d_in_both_tables():
    # El programa del dia abans és el que es té a l'hora del tancament: el del mateix dia
    # és el resultat de la subhasta que es vol descriure, plausible i fora d'abast.
    price = frame_of(range(30))
    scheduled = {"nuclear": frame_of(range(30)) * 7}
    matrices = schedule(price, {}, scheduled)
    (X, _), (rows, _) = matrices.daily, matrices.hourly
    day = X.index[10]
    before = day - pd.Timedelta(days=1)
    for hour, column in enumerate(HOUR_COLUMNS):
        assert X.loc[day, f"nuclear_lag1_{column}"] == scheduled["nuclear"].loc[before, column]
        assert rows.loc[day].iloc[hour]["nuclear_lag1"] == scheduled["nuclear"].loc[before, column]
    assert not any(column.startswith("nuclear_lag1_") for column in rows.columns)
