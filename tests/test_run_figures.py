import datetime
import types

import numpy as np
import pytest

from epf import Refusal, paths
from epf.run import figures, ljung_box, tables

MINUTE = datetime.datetime(2026, 10, 22, 14, 30)


@pytest.fixture
def kinds(tmp_path, monkeypatch):
    """Posa les passades a `tmp_path`, i torna les que es dibuixen."""
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(paths, "SMOKE_OUTPUT", tmp_path / "assaig_output")
    drawn = []
    monkeypatch.setattr(figures, "draw", drawn.append)
    return drawn


def test_each_kind_draws_its_own_run(kinds):
    """Una passada de l'altre tipus semblaria la que es demanava."""
    for smoke in (False, True):
        paths.next_run(smoke, "WITH_PRICE", MINUTE).mkdir(parents=True)
    for smoke in (False, True):
        figures.main(types.SimpleNamespace(smoke=smoke, set_name="WITH_PRICE"))
    assert kinds == [
        paths.next_run(smoke, "WITH_PRICE", MINUTE) for smoke in (False, True)
    ]


def test_no_run_is_refused(kinds):
    with pytest.raises(Refusal, match="^no hi ha cap passada de mesura a "):
        figures.main(types.SimpleNamespace(smoke=False, set_name="WITH_PRICE"))
    assert not kinds


def _family(where, name, saved):
    """Desa a `where` una família de dues seeds, amb els fitxers de les de `saved`."""
    folder = where / name
    paths.write_record(
        folder / paths.FAMILY_RECORD,
        {
            "configurations": {"0": [{"rank": 0}], "1": [{"rank": 0}]},
            "settings": {"SEEDS": [0, 1]},
        },
    )
    for seed in saved:
        np.savez(folder / paths.MODEL.format(seed=seed, rank=0), predicted=np.zeros(1))


def test_a_run_is_drawn_without_its_incomplete_family(tmp_path, monkeypatch):
    """Una família a mitges entrava a les figures com si fos sencera."""
    drawn = []
    monkeypatch.setattr(
        figures,
        "_performance",
        lambda where, described, colours: drawn.append(described),
    )
    monkeypatch.setattr(ljung_box, "report", lambda where, described, colours: None)
    monkeypatch.setattr(tables, "write", lambda where: None)
    _family(tmp_path, "naive_similar_day", saved=(0, 1))
    _family(tmp_path, "naive_previous_day", saved=(0,))
    figures.draw(tmp_path)
    assert [list(described) for described in drawn] == [["naive_similar_day"]]


def test_a_folder_of_no_family_is_left_out_of_the_figures(tmp_path, monkeypatch):
    """La passada de mesura té les carpetes de les DDNN, que el codi ja no té."""
    drawn = []
    monkeypatch.setattr(
        figures,
        "_performance",
        lambda where, described, colours: drawn.append(described),
    )
    monkeypatch.setattr(ljung_box, "report", lambda where, described, colours: None)
    monkeypatch.setattr(tables, "write", lambda where: None)
    _family(tmp_path, "ddnn_jsu", saved=(0, 1))
    _family(tmp_path, "naive_similar_day", saved=(0, 1))
    figures.draw(tmp_path)
    assert [list(described) for described in drawn] == [["naive_similar_day"]]


def test_a_run_with_no_complete_family_is_refused(tmp_path):
    _family(tmp_path, "naive_previous_day", saved=(0,))
    with pytest.raises(Refusal, match="no té cap família completa$"):
        figures.draw(tmp_path)
