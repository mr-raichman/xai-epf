import datetime

import numpy as np
import pandas as pd
import pytest

from epf import Refusal, paths
from epf.contract import matrix, split
from epf.contract.matrix import (
    DAY,
    DAY_OF_WEEK_COLUMNS,
    HOUR,
    build,
    lacking,
    matrices,
    per_hour,
)
from epf.dataset import build as dataset
from epf.information import PRICE, SERIES, WITH_PRICE, resolved
from epf.parameters import HOUR_COLUMNS

# Només el preu, als seus retards.
PREVIOUS_DAY = WITH_PRICE.series[PRICE]
PRICE_ONLY = {PRICE: PREVIOUS_DAY}


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Torna uns preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex(
        [start + datetime.timedelta(days=d) for d in days], name="day"
    )
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def every_series(days=range(30)):
    """Torna cada sèrie de l'information set, el preu per un nombre diferent."""
    return {
        name: frame_of(days) * (number + 1)
        for number, name in enumerate(WITH_PRICE.series)
    }


def test_each_lag_carries_the_right_day():
    price = frame_of(range(30))
    X, y = build({PRICE: price}, PRICE_ONLY)
    day = X.index[10]
    for lag in PREVIOUS_DAY:
        source = price.loc[day - pd.Timedelta(days=lag)]
        for hour in HOUR_COLUMNS:
            assert X.loc[day, f"price_lag{lag}_{hour}"] == source[hour]


def test_the_target_is_the_day_itself():
    price = frame_of(range(30))
    X, y = build({PRICE: price}, PRICE_ONLY)
    assert (y.loc[X.index[10]] == price.loc[X.index[10]]).all()


def test_the_first_complete_row_is_the_longest_lag_away():
    price = frame_of(range(30))
    X, _ = build({PRICE: price}, PRICE_ONLY)
    assert (X.index[0] - price.index[0]).days == max(PREVIOUS_DAY)


def test_a_gap_removes_the_rows_that_need_it_instead_of_shifting_them():
    complete = build({PRICE: frame_of(range(30))}, PRICE_ONLY)[0]
    with_gap = build({PRICE: frame_of([d for d in range(30) if d != 15])}, PRICE_ONLY)[
        0
    ]

    gap = pd.Timestamp("2020-01-16")
    missing = set(complete.index) - set(with_gap.index)
    assert missing == {gap} | {gap + pd.Timedelta(days=lag) for lag in PREVIOUS_DAY}

    # Les files restants segueixen aparellades amb el seu dia.
    day = with_gap.index[-1]
    offset = (day - pd.Timedelta(days=1) - pd.Timestamp("2020-01-01")).days
    assert with_gap.loc[day, "price_lag1_h00"] == offset * 100


def test_the_day_of_week_is_seven_indicators_on_the_delivery_day():
    X, _ = build({PRICE: frame_of(range(30))}, PRICE_ONLY)
    day = X.index[5]
    row = X.loc[day, DAY_OF_WEEK_COLUMNS]
    assert list(row) == [
        1.0 if number == day.dayofweek else 0.0
        for number in range(len(DAY_OF_WEEK_COLUMNS))
    ]
    assert X[DAY_OF_WEEK_COLUMNS].to_numpy().sum(axis=1).tolist() == [1.0] * len(X)


def test_wrong_columns_fail_loudly():
    price = frame_of(range(10)).rename(columns={"h00": "hora0"})
    with pytest.raises(ValueError) as raised:
        build({PRICE: price}, PRICE_ONLY)
    assert not isinstance(raised.value, Refusal)


def test_the_per_hour_row_carries_its_day_and_its_own_hour():
    X, y = build({PRICE: frame_of(range(30))}, PRICE_ONLY)
    rows, target = per_hour(X, y)
    assert len(rows) == len(X) * 24
    day = X.index[10]
    for hour in range(24):
        row = rows.loc[day].iloc[hour]
        assert row[HOUR] == hour
        assert row["price_lag1_h00"] == X.loc[day, "price_lag1_h00"]
        assert target.loc[day].iloc[hour] == y.loc[day, HOUR_COLUMNS[hour]]


def test_day_d_forecasts_sit_on_day_d_in_both_matrices():
    """Una previsió D+1 és del dia que es lliura, i es publica el dia abans.

    Desplaçada un dia, cada fila llegiria la d'ahir, plausible i equivocada.
    """
    series = every_series()
    price, demand = series[PRICE], series["demand_fcst"]
    found = matrices(series, resolved("WITH_PRICE"))
    (X, _), (rows, target) = found.daily, found.hourly
    day = X.index[10]
    for hour, column in enumerate(HOUR_COLUMNS):
        assert X.loc[day, f"demand_fcst_{column}"] == demand.loc[day, column]
        row = rows.loc[day].iloc[hour]
        assert row[HOUR] == hour
        assert row["demand_fcst"] == demand.loc[day, column]
        assert (
            row[f"price_lag1_{column}"] == price.loc[day - pd.Timedelta(days=1), column]
        )
        assert target.loc[day].iloc[hour] == price.loc[day, column]
    assert not any(column.startswith("demand_fcst_") for column in rows.columns)


def test_d_minus_one_schedule_sits_on_day_d_in_both_matrices():
    """El programa del mateix dia és el resultat de la subhasta que es descriu.

    Plausible i fora d'abast: a l'hora del tancament només es té el del dia abans.
    """
    series = every_series()
    nuclear = series["nuclear_sched"]
    found = matrices(series, resolved("WITH_PRICE"))
    (X, _), (rows, _) = found.daily, found.hourly
    day = X.index[10]
    before = day - pd.Timedelta(days=1)
    for hour, column in enumerate(HOUR_COLUMNS):
        assert X.loc[day, f"nuclear_sched_lag1_{column}"] == nuclear.loc[before, column]
        assert (
            rows.loc[day].iloc[hour]["nuclear_sched_lag1"]
            == nuclear.loc[before, column]
        )
    assert not any(column.startswith("nuclear_sched_lag1_") for column in rows.columns)


@pytest.fixture
def downloaded(tmp_path, monkeypatch):
    """Escriu un dataset com `epf dataset`, i torna el fitxer baixat.

    El conjunt és un preu de tres dies que val `0.1 + 0.2` a la primera hora.
    """
    monkeypatch.setattr(paths, "DOWNLOADS", tmp_path / "downloads")
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "DATASET", tmp_path / "data" / "dataset.csv")
    monkeypatch.setattr(paths, "DATASET_RECORD", tmp_path / "data" / "dataset.json")
    path = tmp_path / "downloads" / "price" / "2020.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"indicator": {"values": []}}')
    price = frame_of(range(3)).astype(float)
    price.iloc[0, 0] = 0.1 + 0.2
    dataset.write(dataset.table({PRICE: price}), [path])
    return path


def test_the_dataset_reads_back_exactly_what_was_written(downloaded):
    """Per defecte, pandas hi llegeix 0.3, plausible i no el que es va escriure."""
    assert matrix.series()[PRICE].iloc[0, 0] == 0.1 + 0.2


def test_a_dataset_whose_downloaded_files_changed_is_refused(downloaded):
    matrix.series()
    downloaded.write_text('{"indicator": {"values": [{}]}}')
    with pytest.raises(Refusal, match="desfasat"):
        matrix.series()


def test_a_dataset_that_is_not_the_one_written_is_refused(downloaded):
    with paths.DATASET.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(Refusal, match="ha canviat des que es va fer"):
        matrix.series()


def test_the_stored_matrices_read_back_as_the_ones_the_models_read(tmp_path):
    """Una matriu desada que tornés uns altres decimals no ho diria."""
    # Valors amb tots els decimals d'un float, com els del dataset.
    found = matrices(
        {name: frame / 7 for name, frame in every_series().items()},
        resolved("WITH_PRICE"),
    )
    matrix.write(found, tmp_path)
    for name, (X, y), target in (
        ("daily", found.daily, [f"price_{hour}" for hour in HOUR_COLUMNS]),
        ("hourly", found.hourly, [PRICE]),
    ):
        stored = pd.read_csv(
            tmp_path / paths.MATRIX.format(name=name),
            index_col=DAY,
            parse_dates=[DAY],
            float_precision="round_trip",
        )
        assert list(stored.columns) == list(X.columns) + target
        assert (stored.index == X.index).all()
        assert np.array_equal(stored[X.columns].to_numpy(), X.to_numpy())
        assert np.array_equal(
            stored[target].to_numpy(), np.asarray(y).reshape(len(X), -1)
        )


def test_the_blackout_deletes_its_week_by_the_two_days_that_read_its_schedule():
    """ESIOS no publica el programa de 2025-04-28 12:00 a 2025-04-29 23:00.

    El dia abans, el llegeixen el 29 i el 30, de la setmana del 29 al 5 de maig.
    """
    period = (datetime.date(2025, 4, 22), datetime.date(2025, 5, 12))
    needed = [(PRICE, 0)] + [
        (name, lag) for name, lags in WITH_PRICE.series.items() for lag in lags
    ]
    series = {
        name: frame_of(range(22), start=datetime.date(2025, 4, 21)).astype(float)
        for name in WITH_PRICE.series
    }
    schedule = [name for name in WITH_PRICE.series if SERIES[name].kind == "schedule"]
    for name in schedule:
        series[name].loc["2025-04-28", "h12":] = np.nan
        series[name].loc["2025-04-29"] = np.nan
    days = pd.date_range(*period, freq="D")
    found = split.deleted(period, lacking(series, needed, days))
    assert list(found) == list(pd.date_range("2025-04-29", "2025-05-05").date)
    read = [
        matrix.column(name, lag) for name in schedule for lag in WITH_PRICE.series[name]
    ]
    assert (
        found[datetime.date(2025, 4, 29)] == found[datetime.date(2025, 4, 30)] == read
    )
    assert {found[day] for day in pd.date_range("2025-05-01", "2025-05-05").date} == {
        split.WEEK
    }


def test_a_series_beginning_on_the_period_s_first_day_deletes_its_first_week_at_d_1():
    """Les previsions de Red Eléctrica comencen el primer dia del període.

    Una sèrie en blanc abans del seu primer dia, com la del gas de vapor abans que s'hi
    comptessin zeros, treia de les matrius els dies d'abans sense dir-ho.
    """
    period = (datetime.date(2019, 1, 1), datetime.date(2019, 1, 21))
    series = {
        PRICE: frame_of(range(22), start=datetime.date(2018, 12, 31)),
        "demand_fcst": frame_of(range(21), start=period[0]),
    }
    days = pd.date_range(*period, freq="D")
    found = split.deleted(
        period,
        lacking(
            series,
            [(PRICE, 0), (PRICE, 1), ("demand_fcst", 0), ("demand_fcst", 1)],
            days,
        ),
    )
    assert found == {
        datetime.date(2019, 1, 1): ["demand_fcst_lag1"],
        **{day: split.WEEK for day in pd.date_range("2019-01-02", "2019-01-07").date},
    }


def test_the_reference_is_d_minus_1_s_price_weighed_by_d_minus_1_s_demand_forecast():
    """Pesada per la previsió del dia mateix, o del seu preu, seria plausible."""
    days = pd.date_range("2024-03-04", periods=2, freq="D", name="day")
    price = pd.DataFrame(100.0, index=days, columns=HOUR_COLUMNS)
    price.iloc[0] = 0.0
    price.iloc[0, :2] = (10.0, 30.0)
    weights = pd.DataFrame(1.0, index=days, columns=HOUR_COLUMNS)
    weights.iloc[0] = 0.0
    weights.iloc[0, :2] = (1.0, 3.0)
    assert matrix.references(price, weights).loc[days[1]] == (10 * 1 + 30 * 3) / 4
