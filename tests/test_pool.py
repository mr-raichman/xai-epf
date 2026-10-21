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

    def __init__(self):
        self.advanced, self.failed, self.stopped, self.closed = [], [], [], False

    def start(self, family):
        pass

    def advance(self, family, seed, kind):
        self.advanced.append((family, seed, kind))

    def fail(self, family, seed, message):
        self.failed.append((family, seed, message))

    def stop(self, family, seed, message):
        self.stopped.append((family, seed, message))

    def close(self):
        self.closed = True


def test_a_failing_task_is_reported_and_the_others_still_return(tmp_path):
    """Un error empassat deixaria una família a mitges sense dir-ho."""
    tasks = [
        (("bé", 0, "fit", 0), int, ("3",)),
        (("malament", 1, "trial", 7), int, ("x",)),
        (("també", 2, "fit", 0), int, ("4",)),
    ]
    block, log = Shown(), tmp_path / "workers.log"
    returned, failed = {}, []
    results = pool.done(tasks, 2, 1, log, block, returned.__setitem__, failed.append)
    assert results == returned == {("bé", 0, "fit", 0): 3, ("també", 2, "fit", 0): 4}
    assert failed == [("malament", 1, "trial", 7)]
    assert block.advanced == [("bé", 0, "fit"), ("també", 2, "fit")]
    assert block.failed == [
        ("malament", 1, "ValueError: invalid literal for int() with base 10: 'x'")
    ]
    assert "malament seed 1 trial 7 ha fallat:\nTraceback" in log.read_text(
        encoding="utf-8"
    )
    assert block.closed


def test_a_killed_worker_fails_every_task_drawn_and_no_other_is_read(tmp_path):
    """Un pool trencat feia fallar el `submit` següent, i la passada en un traceback."""
    order = [
        (("a", 0, "fit", 0), int, ("1",)),
        (("b", 1, "fit", 0), os._exit, (1,)),
        (("c", 2, "trial", 0), int, ("2",)),
        (("d", 3, "fit", 0), int, ("3",)),
    ]
    drawn = []

    def tasks():
        for task in order:
            drawn.append(task[0])
            yield task

    block, failed = Shown(), []
    results = pool.done(
        tasks(), 1, 1, tmp_path / "workers.log", block, on_failure=failed.append
    )
    assert drawn == [label for label, *_ in order[:3]]
    assert results == {("a", 0, "fit", 0): 1}
    assert failed == [("b", 1, "fit", 0), ("c", 2, "trial", 0)]
    assert [(family, seed) for family, seed, _ in block.failed] == [("b", 1)]
    assert [(family, seed) for family, seed, _ in block.stopped] == [("c", 2)]
    assert block.closed


def test_a_refusal_in_a_task_is_its_message_alone(tmp_path):
    block, log = Shown(), tmp_path / "workers.log"
    pool.done([(("a", 0, "fit", 2), families.named, (["cap"],))], 1, 1, log, block)
    assert block.failed[0][2].startswith('"cap" no és cap família')
    written = log.read_text(encoding="utf-8")
    assert 'a seed 0 fit 2 ha fallat: "cap" no és cap família' in written
    assert "Traceback" not in written


def test_what_a_worker_writes_goes_to_the_file_and_not_to_the_terminal(tmp_path, capfd):
    # El que escriu una biblioteca de C també, pel descriptor.
    tasks = [
        (("a", 0, "fit", 0), print, ("des de print",)),
        (("b", 0, "fit", 0), os.write, (2, b"des del descriptor\n")),
    ]
    log = tmp_path / "workers.log"
    pool.done(tasks, 2, 1, log, Shown())
    written = log.read_text(encoding="utf-8")
    assert "des de print" in written and "des del descriptor" in written
    assert "des de" not in capfd.readouterr().err


def test_the_thread_cap_is_in_each_worker_s_environment(tmp_path):
    names = list(pool.environment(3))
    tasks = [((name, 0, "fit", 0), os.getenv, (name,)) for name in names]
    results = pool.done(tasks, 2, 3, tmp_path / "workers.log", Shown())
    assert {name: value for (name, *_), value in results.items()} == (
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
            yield ("a", 0, "fit", unit), int, (str(unit),)

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


def test_a_none_task_waits_for_a_return_and_then_every_free_worker_is_fed(tmp_path):
    """Llegint-ne una per tasca tornada, els processos lliures quedarien aturats."""
    read, returned = [], set()

    def tasks():
        read.append("a")
        yield ("a", 0, "trial", 0), int, ("1",)
        while "a" not in returned:
            yield None
        for unit in ("b", "c"):
            read.append(unit)
            yield (unit, 0, "fit", 0), int, ("1",)

    seen = {}

    def on_return(label, result):
        returned.add(label[0])
        seen[label[0]] = len(read)

    pool.done(tasks(), 2, 1, tmp_path / "workers.log", Shown(), on_return=on_return)
    assert seen == {"a": 1, "b": 3, "c": 3}
