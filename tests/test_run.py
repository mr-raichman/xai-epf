import datetime
import sys
import types

import pandas as pd
import pytest

from epf import Refusal, parameters, paths
from epf.contract import matrix, record
from epf.run import families, figures, pool, run, search


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Prepara passades d'assaig de dues famílies d'un model que no ajusta res.

    Cada família desa un fitxer buit per configuració del pla que rep, i falla si és a
    `failing`. Les tasques es fan aquí mateix i no en processos nous, que no veurien res
    del que es canvia aquí. Torna l'estat, amb `main`, que en fa una.
    """
    state = types.SimpleNamespace(
        calls=[], failing=set(), fingerprints={"code": "codi", "dataset": "conjunt"}
    )

    def train(family, folder, configurations, explained_days, saved):
        state.calls.append((family, configurations))
        if family in state.failing:
            raise RuntimeError("interrompuda")
        for seed, planned in configurations.items():
            for one in planned:
                (folder / paths.SEED.format(seed=seed, rank=one["rank"])).touch()
                saved()
        return [
            [seed, one["rank"]]
            for seed, planned in configurations.items()
            for one in planned
        ]

    module = types.ModuleType("fake_trainer")
    module.FAMILIES = {
        name: {"fixed": {}, "placeholder": {"value": 1}, "kept": 1, "searched_as": name}
        for name in ("first", "second")
    }
    module.train = train
    monkeypatch.setitem(sys.modules, "fake_trainer", module)
    monkeypatch.setattr(
        run, "TRAINERS", {"first": "fake_trainer", "second": "fake_trainer"}
    )
    monkeypatch.setattr(families, "EVERY", frozenset({"first", "second"}))
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "SEARCHES", tmp_path / "searches")
    built = types.SimpleNamespace(daily=(pd.DataFrame({"x": [0.0]}), None))
    monkeypatch.setattr(matrix, "series", lambda: None)
    monkeypatch.setattr(matrix, "matrices", lambda series: built)
    monkeypatch.setattr(matrix, "write", lambda found, folder: None)
    monkeypatch.setattr(record, "fingerprints", lambda: dict(state.fingerprints))
    monkeypatch.setattr(
        search, "searcher", lambda family: types.SimpleNamespace(MATRIX="daily")
    )
    monkeypatch.setattr(figures, "draw", lambda where: None)

    def done(tasks, workers, threads, log, block):
        results, failed = {}, []
        for label, function, arguments in tasks:
            try:
                results[label] = function(*arguments)
            except RuntimeError:
                failed.append(label)
        return results, failed

    monkeypatch.setattr(pool, "done", done)
    # Una sola seed, sigui quina sigui la variant d'assaig que tria `chosen`.
    real = parameters.chosen
    monkeypatch.setattr(
        parameters, "chosen", lambda smoke: real(smoke) | {"SEEDS": (0,)}
    )
    state.module = module
    state.main = lambda: run.main(types.SimpleNamespace(smoke=True, family=None))
    return state


def _name(number):
    """Torna el nom de la passada d'assaig d'avui que porta el número `number`."""
    return f"{paths.SMOKE_PREFIX}{datetime.date.today().isoformat()}_{number:02d}"


def test_a_killed_run_is_continued_each_family_on_the_plan_it_recorded(fake, capsys):
    """Refet en continuar, el pla canviaria els números de les famílies que falten."""
    # La segona família falla la primera vegada, i el seu número de lloc canvia abans de
    # continuar; la primera, ja sencera, no es torna a ajustar.
    fake.failing = {"second"}
    assert fake.main() == 1
    fake.failing = set()
    fake.module.FAMILIES["second"]["placeholder"] = {"value": 2}
    assert fake.main() == 0
    assert [family for family, _ in fake.calls] == ["first", "second", "second"]
    assert fake.calls[-1][1] == {
        0: [{"rank": 0, "numbers": {"value": 1}, "unsearched": True}]
    }
    shown = capsys.readouterr().err
    assert f"Passada {_name(1)}\n" in shown
    assert f"Passada {_name(1)}, continuada\n" in shown


def test_a_finished_run_is_never_continued(fake, capsys):
    """Una passada acabada que es continués en barrejaria dues."""
    assert fake.main() == 0
    assert fake.main() == 0
    shown = capsys.readouterr().err
    assert f"Passada {_name(1)}\n" in shown and f"Passada {_name(2)}\n" in shown


def test_a_run_made_by_other_code_is_refused_and_not_continued(fake):
    fake.failing = {"second"}
    assert fake.main() == 1
    fake.failing = set()
    fake.fingerprints["code"] = "un altre codi"
    calls = len(fake.calls)
    with pytest.raises(Refusal, match=f"la passada {_name(1)} .* un altre codi;"):
        fake.main()
    assert len(fake.calls) == calls
