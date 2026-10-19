import os

import pytest

from epf.run import families, pool


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    """Deixa l'entorn d'aquest procés com era: `done` hi posa el límit."""
    for name in pool.environment(1):
        monkeypatch.setenv(name, "")


class Shown:
    """Un bloc que només recorda el que se li diu."""

    unit = "fit"

    def __init__(self):
        self.advanced, self.failed, self.closed = [], [], False

    def start(self, family):
        pass

    def advance(self, family, seed):
        self.advanced.append((family, seed))

    def fail(self, family, seed, message):
        self.failed.append((family, seed, message))

    def close(self):
        self.closed = True


def test_a_failing_task_is_reported_and_the_others_still_return(tmp_path):
    """Un error empassat deixaria una família a mitges sense dir-ho."""
    tasks = [
        (("bé", 0, 0), int, ("3",)),
        (("malament", 1, 0), int, ("x",)),
        (("també", 2, 0), int, ("4",)),
    ]
    block, log = Shown(), tmp_path / "workers.log"
    returned, failed = {}, []
    results = pool.done(tasks, 2, 1, log, block, returned.__setitem__, failed.append)
    assert results == returned == {("bé", 0, 0): 3, ("també", 2, 0): 4}
    assert failed == [("malament", 1, 0)]
    assert block.failed == [
        ("malament", 1, "ValueError: invalid literal for int() with base 10: 'x'")
    ]
    assert "malament seed 1 fit 0 ha fallat:\nTraceback" in log.read_text(
        encoding="utf-8"
    )
    assert block.closed


def test_a_refusal_in_a_task_is_its_message_alone(tmp_path):
    block, log = Shown(), tmp_path / "workers.log"
    pool.done([(("x", 0, 0), families.named, (["cap"], {}))], 1, 1, log, block)
    assert block.failed[0][2].startswith('"cap" no és cap família')
    assert "ha fallat" not in log.read_text(encoding="utf-8")


def test_what_a_worker_writes_goes_to_the_file_and_not_to_the_terminal(tmp_path, capfd):
    # El que escriu una biblioteca de C també, pel descriptor.
    tasks = [
        (("a", 0, 0), print, ("des de print",)),
        (("b", 0, 0), os.write, (2, b"des del descriptor\n")),
    ]
    log = tmp_path / "workers.log"
    pool.done(tasks, 2, 1, log, Shown())
    written = log.read_text(encoding="utf-8")
    assert "des de print" in written and "des del descriptor" in written
    assert "des de" not in capfd.readouterr().err


def test_the_thread_cap_is_in_each_worker_s_environment(tmp_path):
    names = list(pool.environment(3))
    tasks = [((name, 0, 0), os.getenv, (name,)) for name in names]
    results = pool.done(tasks, 2, 3, tmp_path / "workers.log", Shown())
    assert {name: value for (name, _, _), value in results.items()} == (
        pool.environment(3)
    )


def test_the_tasks_are_read_only_as_a_worker_frees(tmp_path):
    """Llegides totes d'entrada, una cerca en demanaria tots els trials alhora.

    Un Ctrl+C els deixaria demanats i sense fer.
    """
    read = []

    def tasks():
        for unit in range(4):
            read.append(unit)
            yield ("a", 0, unit), int, (str(unit),)

    seen = []
    pool.done(
        tasks(),
        2,
        1,
        tmp_path / "workers.log",
        Shown(),
        on_return=lambda label, result: seen.append(len(read)),
    )
    assert seen[0] == 2 and read == [0, 1, 2, 3]
