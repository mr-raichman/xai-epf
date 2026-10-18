"""Els processos que fan les tasques: una tasca que peta no s'emporta les altres, i el
límit de fils arriba a l'entorn de cada procés.

Res d'això no peta quan falla. Un error empassat deixaria una família a mitges sense
dir-ho, i un límit que no arribés faria cada ajust amb tots els fils de la màquina, i
amb uns altres números que els que dóna el límit.
"""

import os

import pytest

from epf.run import pool


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    """L'entorn d'aquest procés com era, en acabar: `done` hi posa el límit."""
    for name in pool.environment(1):
        monkeypatch.setenv(name, "")


def test_a_failing_task_is_reported_and_the_others_still_return(capsys):
    tasks = [("bé", int, ("3",)), ("malament", int, ("x",)), ("també", int, ("4",))]
    results, failed = pool.done(tasks, 2, 1)
    assert results == {"bé": 3, "també": 4}
    assert failed == ["malament"]
    assert "malament: ha petat" in capsys.readouterr().out


def test_the_thread_cap_is_in_each_worker_s_environment():
    names = list(pool.environment(3))
    results, _ = pool.done([(name, os.getenv, (name,)) for name in names], 2, 3)
    assert results == pool.environment(3)
