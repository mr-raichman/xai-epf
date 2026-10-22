import datetime
import types

import pytest

from epf import Refusal, paths
from epf.run import families, figures

MINUTE = datetime.datetime(2026, 10, 22, 14, 30)


@pytest.fixture
def kinds(tmp_path, monkeypatch):
    """Posa les passades a `tmp_path`, i torna les que es dibuixen."""
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(paths, "SMOKE_OUTPUT", tmp_path / "assaig_output")
    drawn = []
    monkeypatch.setattr(figures, "draw", drawn.append)
    return drawn


def test_each_kind_draws_its_own_run(kinds, monkeypatch):
    """Una passada de l'altre tipus semblaria la que es demanava."""
    for smoke in (False, True):
        paths.next_run(smoke, MINUTE).mkdir(parents=True)
    monkeypatch.setattr(families, "finished", lambda where: True)
    figures.main(types.SimpleNamespace(smoke=False))
    figures.main(types.SimpleNamespace(smoke=True))
    assert kinds == [paths.next_run(False, MINUTE), paths.next_run(True, MINUTE)]


def test_no_run_or_an_unfinished_one_is_refused(kinds, monkeypatch):
    with pytest.raises(Refusal, match="^no hi ha cap passada de mesura a "):
        figures.main(types.SimpleNamespace(smoke=False))
    paths.next_run(False, MINUTE).mkdir(parents=True)
    monkeypatch.setattr(families, "finished", lambda where: False)
    with pytest.raises(Refusal, match="^la passada 2026-10-22_14-30 no és acabada$"):
        figures.main(types.SimpleNamespace(smoke=False))
    assert not kinds
