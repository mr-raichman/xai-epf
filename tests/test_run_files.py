"""Escriure un dia cada vegada i reprendre'n el fil.

El camí endavant escriu les prediccions dia a dia, de manera que el directori d'una
família és el magatzem d'on es reprèn. Això només val si una fila afegida surt igual
que la mateixa fila dins del `to_csv` de la taula sencera: si no, una execució represa
donaria un fitxer diferent d'una de seguida i un diff entre una reconstrucció i
l'execució deixaria de voler dir res.
"""
import datetime

import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.results import run_files

ONE, OTHER = "one", "other"


def frame_of(days, start=datetime.date(2024, 1, 1), offset=0.0):
    index = pd.DatetimeIndex([start + datetime.timedelta(days=day)
                              for day in range(days)], name="day")
    rows = [[day * 100 + hour + offset for hour in range(24)] for day in range(days)]
    return pd.DataFrame(rows, index=index, columns=HOUR_COLUMNS)


def appended(folder, frames):
    """Escriu les taules dia a dia, com ho fa el camí endavant."""
    folder.mkdir(parents=True, exist_ok=True)
    for position in range(len(next(iter(frames.values())))):
        run_files.append_forecasts(folder, {
            name: frame.iloc[position:position + 1] for name, frame in frames.items()})


def test_a_day_at_a_time_gives_the_same_bytes_as_the_whole_table(tmp_path):
    # Els valors que trenquen un format curt: el zero negatiu, un decimal periòdic,
    # un número molt petit i un de nou xifres.
    frame = frame_of(5)
    frame.iloc[0, 0] = -0.0
    frame.iloc[0, 1] = 1 / 3
    frame.iloc[0, 2] = 1e-17
    frame.iloc[0, 3] = 123456789.123456789

    whole = tmp_path / "whole"
    whole.mkdir()
    run_files.save_forecasts(whole, {ONE: frame})
    appended(tmp_path / "day_by_day", {ONE: frame})

    name = run_files.FORECASTS.format(name=ONE)
    assert (tmp_path / "day_by_day" / name).read_bytes() == (whole / name).read_bytes()


def test_what_was_appended_reads_back_exactly(tmp_path):
    # Les mètriques es calculen de les prediccions rellegides del disc, de manera que
    # un bit perdut en tornar-les a llegir mouria les xifres d'una execució represa.
    frame = frame_of(4)
    frame.iloc[0, 0] = 1 / 3
    appended(tmp_path / "family", {ONE: frame})
    back = run_files.load_forecasts(tmp_path / "family")[ONE]
    assert (back.to_numpy() == frame.to_numpy()).all()
    assert list(back.index) == list(frame.index)


def test_the_resume_day_is_the_last_one_every_forecast_has(tmp_path):
    # Una caiguda entre dos fitxers deixa un pronòstic més llarg que l'altre. Manda
    # el curt: continuar des del llarg alinearia el curt amb els dies equivocats.
    folder = tmp_path / "family"
    appended(folder, {ONE: frame_of(5), OTHER: frame_of(5)})
    run_files.append_forecasts(folder, {ONE: frame_of(6).iloc[5:6]})
    assert run_files.last_common_day(folder) == pd.Timestamp("2024-01-05")


def test_dropping_after_the_resume_day_leaves_them_all_at_it(tmp_path):
    folder = tmp_path / "family"
    appended(folder, {ONE: frame_of(5), OTHER: frame_of(5)})
    run_files.append_forecasts(folder, {ONE: frame_of(6).iloc[5:6]})
    run_files.drop_after(folder, pd.Timestamp("2024-01-05"))
    forecasts = run_files.load_forecasts(folder)
    assert {frame.index[-1] for frame in forecasts.values()} == {pd.Timestamp("2024-01-05")}
    assert run_files.last_common_day(folder) == pd.Timestamp("2024-01-05")


def test_a_directory_with_nothing_in_it_has_no_resume_day(tmp_path):
    assert run_files.last_common_day(tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    assert run_files.last_common_day(tmp_path / "empty") is None
