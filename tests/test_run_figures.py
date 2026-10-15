"""Les figures d'una passada sense nom: les de l'última acabada del seu tipus.

Res no peta si s'equivoca: dibuixaria una passada a mig fer, o la de l'altre tipus, i les
figures semblarien les que es demanaven.
"""
import types

from epf import paths
from epf.run import families, figures


def test_with_no_name_the_latest_finished_run_of_its_kind_is_drawn(tmp_path, monkeypatch):
    # La més nova de mesura és a mig fer, i l'assaig, d'un altre comptador, en té una de
    # més nova que cap de mesura.
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    smoke = paths.SMOKE_PREFIX
    for name in ("2026-10-14_01", "2026-10-15_01", "2026-10-15_02", f"{smoke}2026-10-15_03"):
        (tmp_path / name).mkdir()
    done = {"2026-10-14_01", "2026-10-15_01", f"{smoke}2026-10-15_03"}
    monkeypatch.setattr(families, "finished", lambda where: where.name in done)
    drawn = []
    monkeypatch.setattr(figures, "draw", lambda where: drawn.append(where.name))
    assert figures.main(types.SimpleNamespace(run=None, smoke=False)) == 0
    assert figures.main(types.SimpleNamespace(run=None, smoke=True)) == 0
    assert drawn == ["2026-10-15_01", f"{smoke}2026-10-15_03"]


def test_with_no_finished_run_of_its_kind_nothing_is_drawn(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "2026-10-15_01").mkdir()
    monkeypatch.setattr(families, "finished", lambda where: False)
    monkeypatch.setattr(figures, "draw", lambda where: None)
    assert figures.main(types.SimpleNamespace(run=None, smoke=False)) == 1
