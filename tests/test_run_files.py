"""Escriure un dia cada vegada i reprendre'n el fil.

El camí endavant escriu les prediccions dia a dia, de manera que el directori d'una
família és el magatzem d'on es reprèn. Això només val si una fila afegida surt igual
que la mateixa fila dins del `to_csv` de la taula sencera: si no, una execució represa
donaria un fitxer diferent d'una de seguida i un diff entre una reconstrucció i
l'execució deixaria de voler dir res.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

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


def test_the_distribution_round_trips_day_by_day(tmp_path):
    # S'hi afegeix un dia cada vegada com a les prediccions, perquè el camí endavant
    # es reprèn. Si es desés sencer al final, una execució represa n'hauria desat
    # només els dies de l'última sessió.
    folder = tmp_path / "family"
    folder.mkdir()
    names = ("location", "scale")
    rng = np.random.default_rng(0)
    written = {}
    for day in frame_of(4).index:
        written[day] = rng.normal(60, 5, size=(24, len(names)))
        run_files.append_distribution(folder, ONE, day, written[day], names)

    days, parameters, found = run_files.load_distribution(folder, ONE)
    assert found == list(names)
    assert list(days) == list(written)
    assert parameters.shape == (4, 24, 2)
    assert np.allclose(parameters[0], written[days[0]], atol=1e-6)


def test_dropping_after_a_day_truncates_the_distribution_too(tmp_path):
    # Es talla amb les prediccions: si no, en reprendre quedaria més llarga que elles
    # i el CRPS es calcularia sobre dies que les prediccions ja no tenen.
    folder = tmp_path / "family"
    appended(folder, {ONE: frame_of(5)})
    for day in frame_of(5).index:
        run_files.append_distribution(folder, ONE, day, np.zeros((24, 1)), ("scale",))

    run_files.drop_after(folder, pd.Timestamp("2024-01-03"))
    days, _, _ = run_files.load_distribution(folder, ONE)
    assert days[-1] == pd.Timestamp("2024-01-03")


def test_hours_out_of_order_in_the_distribution_fail_loudly(tmp_path):
    # Les files es remunten per ordre, de manera que una fila fora de lloc faria
    # comparar una hora amb el preu d'una altra sense que res petés.
    folder = tmp_path / "family"
    folder.mkdir()
    path = folder / run_files.DISTRIBUTION.format(name=ONE)
    path.write_text("day,hour,scale\n2024-01-01,1,5.0\n2024-01-01,0,5.0\n")
    with pytest.raises(ValueError, match="ordre"):
        run_files.load_distribution(folder, ONE)


def test_a_directory_with_nothing_in_it_has_no_resume_day(tmp_path):
    assert run_files.last_common_day(tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    assert run_files.last_common_day(tmp_path / "empty") is None
