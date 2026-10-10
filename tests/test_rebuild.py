"""Que una reconstrucció no perdi el CRPS, que és el que va passar.

`metrics` no el calcula: no surt de les prediccions sinó dels paràmetres que hi ha al
costat. Mesurat el 2026-10-10, una reconstrucció d'una execució distribucional
reescrivia `metrics.csv` sense aquella fila i no es queixava, de manera que el fitxer
quedava més pobre que el que havia escrit el camí endavant i la propietat que tot el
que es deriva d'una execució es pot refer deixava de ser certa.
"""
import datetime

import numpy as np
import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.models.ddnn.distributions import NORMAL
from epf.rebuild import distribution_rows
from epf.results import run_files

NAME = "ddnn_normal"
DAYS = 5


def written(folder):
    """Una execució distribucional al disc, tal com el camí endavant la deixa."""
    folder.mkdir(parents=True)
    index = pd.DatetimeIndex([datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
                              for day in range(DAYS)], name="day")
    actual = pd.DataFrame(60.0, index=index, columns=HOUR_COLUMNS)
    run_files.save_forecasts(folder, {NAME: actual})
    for day in index:
        run_files.append_distribution(folder, NAME, day,
                                      np.column_stack([np.full(24, 60.0),
                                                       np.full(24, 5.0)]),
                                      NORMAL.names)
    run_files.save_record(folder, {"family": NAME, "members": [NAME],
                                   "headlines": [NAME]})
    return actual


def test_a_distributional_run_gets_its_crps_row_back(tmp_path):
    actual = written(tmp_path / NAME)
    rows = distribution_rows(tmp_path / NAME, actual)
    assert list(rows["metric"]) == ["crps"]
    assert rows["forecast"].iloc[0] == NAME
    assert rows["value"].iloc[0] > 0


def test_a_family_that_predicts_prices_adds_no_rows(tmp_path):
    # Les quatre que no prediuen cap distribució no han de guanyar cap mètrica, i una
    # família que el registre no reconegui tampoc: és el cas del directori de la
    # combinació, que hi arriba com qualsevol altre.
    folder = tmp_path / "lear"
    folder.mkdir(parents=True)
    index = pd.DatetimeIndex([datetime.date(2024, 1, 1)], name="day")
    actual = pd.DataFrame(60.0, index=index, columns=HOUR_COLUMNS)
    run_files.save_forecasts(folder, {"lear": actual})
    run_files.save_record(folder, {"family": "lear"})
    assert distribution_rows(folder, actual) is None

    run_files.save_record(folder, {"members": ["a", "b"]})
    assert distribution_rows(folder, actual) is None
