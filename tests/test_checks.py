"""Les comprovacions del conjunt de dades, les mateixes per a totes les sèries.

Cadascuna és d'un error que produiria un número plausible més endavant en lloc de petar:
un dia i una hora repetits doblen el seu pes al calibratge, unes files fora d'ordre o un
dia que falta desplacen els retards cap a un dia que no és el que toca, un dia sense les
seves 24 hores desalinea les hores, i un valor absent es propaga al model.
"""

import pandas as pd
import pytest

from epf.dataset.check.checks import check

DAYS = 10
NONE = pd.DatetimeIndex([])


def table_of(days=DAYS, names=("price",)):
    """Un conjunt de dades coherent, de `days` dies des del 2024-01-01."""
    index = pd.MultiIndex.from_tuples(
        [
            (day.strftime("%Y-%m-%d"), hour)
            for day in pd.date_range("2024-01-01", periods=days, freq="D")
            for hour in range(24)
        ],
        names=["day", "hour"],
    )
    return pd.DataFrame({name: 50.0 for name in names}, index=index)


def test_a_coherent_dataset_passes():
    check(table_of(), {"price": NONE})


def test_a_repeated_day_and_hour_is_refused():
    table = table_of()
    doubled = pd.concat([table, table.iloc[[30]]]).sort_index()
    with pytest.raises(ValueError, match="repeteix un dia i una hora"):
        check(doubled, {"price": NONE})


def test_unsorted_rows_are_refused():
    with pytest.raises(ValueError, match="ordenades"):
        check(table_of().iloc[::-1], {"price": NONE})


def test_a_day_missing_from_the_middle_is_refused_and_named():
    table = table_of()
    gap = table.drop("2024-01-05", level="day")
    with pytest.raises(ValueError, match="2024-01-05"):
        check(gap, {"price": NONE})


def test_a_day_without_its_24_hours_is_refused():
    table = table_of()
    with pytest.raises(ValueError, match="1 dia sense 24 hores, el 2024-01-03"):
        check(table.drop(("2024-01-03", 23)), {"price": NONE})


def test_one_absent_value_is_refused_and_its_series_named():
    table = table_of(names=("price", "demand_fcst"))
    table.iloc[50, 1] = None
    with pytest.raises(ValueError, match="la sèrie demand_fcst té 1 valor absent"):
        check(table, {"price": NONE, "demand_fcst": NONE})


def test_only_the_declared_gap_may_be_absent():
    # Les hores de l'apagada, que ESIOS no publica al programa, s'admeten perquè la
    # sèrie les declara; una hora absent a qualsevol altre lloc segueix refusada, també
    # al costat.
    table = table_of()
    gap = pd.date_range("2024-01-03 12:00", "2024-01-04 23:00", freq="h")
    table.loc[("2024-01-03", slice(12, 23)), "price"] = None
    table.loc[("2024-01-04", slice(None)), "price"] = None
    check(table, {"price": gap})
    table.loc[("2024-01-03", 11), "price"] = None
    with pytest.raises(ValueError, match="2024-01-03 11:00"):
        check(table, {"price": gap})
