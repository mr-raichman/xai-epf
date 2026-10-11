"""Que l'atribució desada reprodueixi la predicció que té al costat."""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.attribution.stored import (ONE, PRICE_UNITS, TRANSFORMED_UNITS, Attribution,
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
                        np.full((1, 24), base), None, None, PRICE_UNITS)
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


def ensemble(folder, base, contributions, transformed, members=(10.0, 20.0, 30.0),
             without=None):
    """Un conjunt que atribueix en un espai transformat, amb tres finestres.

    `members` són les seves prediccions en EUR/MWh, que és el que ha de tornar el CSV,
    i `transformed` les mateixes al seu espai, que és contra el que es comprova
    l'additivitat. `without` és el camp que l'`attribute` s'ha deixat de desar.
    """
    folder.mkdir(parents=True, exist_ok=True)
    count = len(members)
    kept = [Attribution(tuple("abc"[:count]), ["x", "y"],
                        np.full((count, 2, 24), contributions),
                        np.full((count, 24), base),
                        np.repeat(np.array(members)[:, None], 24, axis=1),
                        np.full((count, 24), transformed), TRANSFORMED_UNITS)
            for _ in range(DAYS)]
    if without:
        kept = [one._replace(**{without: None}) for one in kept]
    frame = frame_of(np.mean(members))
    save(folder, "lear", frame.index, kept)
    return frame


def test_an_ensemble_is_checked_against_the_mean_of_its_members(tmp_path):
    # Les contribucions a l'espai transformat no es poden comparar amb el CSV; d'una
    # família de diversos membres, qui ha de reproduir-lo és la mitjana dels membres.
    folder = tmp_path / "lear"
    frame = ensemble(folder, base=0.4, contributions=0.1, transformed=0.6)
    assert refuse_inconsistent(folder, "lear", frame) < 1e-3
    with pytest.raises(ValueError, match="mitjana de les prediccions"):
        refuse_inconsistent(folder, "lear", frame_of(0.0))


def test_an_ensemble_whose_base_does_not_add_up_fails_loudly(tmp_path):
    # Mentre la predicció al mateix espai no es desava, aquesta era la família amb
    # l'atribució exacta per construcció i l'única que no comprovava res: la base de
    # cada finestra podia ser qualsevol i la mitjana en EUR/MWh seguia quadrant.
    folder = tmp_path / "lear"
    frame = ensemble(folder, base=0.4, contributions=0.1, transformed=99.0)
    with pytest.raises(ValueError, match="no és additiva"):
        refuse_inconsistent(folder, "lear", frame)


def test_a_transformed_attribution_that_leaves_out_what_checks_it_is_refused(tmp_path):
    # Sense les prediccions en EUR/MWh no hi ha res a comparar amb el CSV, i sense les
    # del seu espai no hi ha res contra què sumar: totes dues comprovacions passarien
    # sense haver-se fet.
    for without, message in (("forecasts", "prediccions dels membres"),
                             ("transformed", "additivitat")):
        folder = tmp_path / without
        frame = ensemble(folder, base=0.4, contributions=0.1, transformed=0.6,
                         without=without)
        with pytest.raises(ValueError, match=message):
            refuse_inconsistent(folder, "lear", frame)
