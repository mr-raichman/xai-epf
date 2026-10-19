import datetime

import numpy as np
import pandas as pd
import pytest

from epf import paths
from epf.contract import matrix
from epf.contract.matrix import (
    DAY,
    DAY_OF_WEEK_COLUMNS,
    HOUR,
    INFORMATION,
    PRICE,
    build,
    matrices,
    per_hour,
)
from epf.dataset import build as dataset
from epf.parameters import HOUR_COLUMNS, PREVIOUS_DAY

# Només el preu, als seus retards.
PRICE_ONLY = {PRICE: PREVIOUS_DAY}


def frame_of(days, start=datetime.date(2020, 1, 1)):
    """Torna uns preus sintètics on cada valor és dia * 100 + hora."""
    index = pd.DatetimeIndex(
        [start + datetime.timedelta(days=d) for d in days], name="day"
    )
    rows = [[d * 100 + hour for hour in range(24)] for d in days]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def every_series(days=range(30)):
    """Torna cada sèrie del conjunt d'informació, el preu per un nombre diferent."""
    return {
        name: frame_of(days) * (number + 1) for number, name in enumerate(INFORMATION)
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
    with pytest.raises(ValueError):
        build({PRICE: price}, PRICE_ONLY)


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
    found = matrices(series)
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
    found = matrices(series)
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
    with pytest.raises(ValueError, match="desfasat"):
        matrix.series()


def test_a_dataset_that_is_not_the_one_written_is_refused(downloaded):
    with paths.DATASET.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="ha canviat des que es va fer"):
        matrix.series()


def test_the_stored_matrices_read_back_as_the_ones_the_models_read(tmp_path):
    """Una matriu desada que tornés uns altres decimals no ho diria."""
    # Valors amb tots els decimals d'un float, com els del dataset.
    found = matrices({name: frame / 7 for name, frame in every_series().items()})
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
