"""Que l'atribució desada reprodueixi la predicció que té al costat."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.models.attribution import (ONE, PRICE_UNITS, Attribution,
                                    refuse_inconsistent, save)

DAYS = 3


def frame_of(values):
    index = pd.DatetimeIndex([datetime.date(2024, 1, 1) + datetime.timedelta(days=d)
                              for d in range(DAYS)], name="day")
    return pd.DataFrame(values, index=index, columns=HOUR_COLUMNS)


def tree(folder, base, contributions):
    """Una família que atribueix en EUR/MWh: base més contribucions és la predicció."""
    folder.mkdir(parents=True, exist_ok=True)
    kept = [Attribution(ONE, ["a", "b"], np.full((1, 2, 24), contributions),
                        np.full((1, 24), base), None, PRICE_UNITS)
            for _ in range(DAYS)]
    frame = frame_of(base + 2 * contributions)
    save(folder, "tree", frame.index, kept)
    return frame


def test_a_consistent_attribution_passes_and_says_how_close(tmp_path):
    frame = tree(tmp_path, base=50.0, contributions=1.5)
    worst = refuse_inconsistent(tmp_path, "tree", frame)
    # Es desa en float32, de manera que el residu no és zero i sí molt petit.
    assert worst < 1e-3


def test_an_attribution_that_does_not_add_up_fails_loudly(tmp_path):
    # La predicció desada no és la que donen la base i les contribucions: és
    # exactament el que la comprovació existeix per agafar, i sense ella el
    # directori semblaria sencer.
    tree(tmp_path, base=50.0, contributions=1.5)
    with pytest.raises(ValueError, match="no reprodueix"):
        refuse_inconsistent(tmp_path, "tree", frame_of(99.0))


def test_an_ensemble_is_checked_against_the_mean_of_its_members(tmp_path):
    # Les contribucions de LEAR són a l'espai transformat i no es poden comparar
    # amb el CSV; qui ha de reproduir-lo és la mitjana de les seves finestres.
    folder = tmp_path / "lear"
    folder.mkdir(parents=True)
    members = np.array([10.0, 20.0, 30.0])
    kept = [Attribution(("a", "b", "c"), ["x"], np.zeros((3, 1, 24)), np.zeros((3, 24)),
                        np.repeat(members[:, None], 24, axis=1), "espai transformat")
            for _ in range(DAYS)]
    frame = frame_of(members.mean())
    save(folder, "lear", frame.index, kept)
    assert refuse_inconsistent(folder, "lear", frame) < 1e-3
    with pytest.raises(ValueError, match="mitjana de les prediccions"):
        refuse_inconsistent(folder, "lear", frame_of(0.0))
