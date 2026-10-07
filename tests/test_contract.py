"""El contracte que qualsevol conjunt de dades ha de complir.

Les cinc condicions són les que produeixen un número plausible més endavant en lloc
de petar: un dia repetit dobla el seu pes al calibratge, un dia fora d'ordre o un
forat desplacen els retards cap a un dia que no és el que toca, una graella que no
és d'un dia sencer desalinea les hores, i un valor absent es propaga al model.
"""
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.data.contract import refuse_incoherent

DAYS = 10


def frame_of(days=DAYS, columns=HOUR_COLUMNS):
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=days, freq="D"), name="day")
    return pd.DataFrame(50.0, index=index, columns=list(columns))


def test_a_coherent_dataset_passes_and_comes_back():
    frame = frame_of()
    assert refuse_incoherent("price", frame) is frame


def test_a_repeated_day_is_refused():
    frame = frame_of()
    doubled = pd.concat([frame, frame.iloc[[3]]]).sort_index()
    with pytest.raises(ValueError, match="repetits"):
        refuse_incoherent("price", doubled)


def test_an_unsorted_index_is_refused():
    frame = frame_of()
    with pytest.raises(ValueError, match="ordenat"):
        refuse_incoherent("price", frame.iloc[::-1])


def test_a_day_missing_from_the_middle_is_refused_and_named():
    frame = frame_of()
    gap = frame.drop(frame.index[4])
    with pytest.raises(ValueError, match="2024-01-05"):
        refuse_incoherent("price", gap)


def test_a_grid_that_is_not_a_whole_day_is_refused():
    with pytest.raises(ValueError, match="23 columnes"):
        refuse_incoherent("price", frame_of(columns=HOUR_COLUMNS[:-1]))


def test_one_absent_value_is_refused():
    frame = frame_of()
    frame.iloc[2, 5] = None
    with pytest.raises(ValueError, match="1 valors absents"):
        refuse_incoherent("price", frame)


def test_the_dataset_name_is_in_the_message():
    with pytest.raises(ValueError, match="el conjunt demanda no és coherent"):
        refuse_incoherent("demanda", frame_of(columns=HOUR_COLUMNS[:-1]))
