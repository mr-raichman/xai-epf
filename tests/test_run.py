"""Una passada: quina es continua, quin pla ajusta cada família, i quina es refusa.

Res d'això no peta quan falla. Una passada acabada que es continués en barrejaria dues; un
pla fet de nou a mitja passada ajustaria unes famílies amb uns números i, en tornar-hi,
les altres amb uns altres; i una passada continuada amb un altre codi tindria famílies
que no ha fet el mateix codi.
"""
import datetime
import sys
import types

import pandas as pd
import pytest

from epf import parameters, paths
from epf.contract import matrix, record
from epf.run import families, figures, run, search


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Passades d'assaig de dues famílies d'un model que no ajusta res: cada una desa un
    fitxer buit per cada configuració del pla que rep, i peta si és a `failing`."""
    state = types.SimpleNamespace(calls=[], failing=set(),
                                  fingerprints={"code": "codi", "dataset": "conjunt"})

    def train(family, folder, configurations, explained_days):
        state.calls.append((family, configurations))
        if family in state.failing:
            raise RuntimeError("interrompuda")
        for seed, planned in configurations.items():
            for one in planned:
                (folder / paths.SEED.format(seed=seed, rank=one["rank"])).touch()
        return [[seed, one["rank"]] for seed, planned in configurations.items()
                for one in planned]

    module = types.ModuleType("fake_trainer")
    module.FAMILIES = {name: {"fixed": {}, "placeholder": {"value": 1}, "kept": 1,
                              "searched_as": name} for name in ("first", "second")}
    module.train = train
    monkeypatch.setitem(sys.modules, "fake_trainer", module)
    monkeypatch.setattr(run, "TRAINERS", {"first": "fake_trainer", "second": "fake_trainer"})
    monkeypatch.setattr(families, "EVERY", frozenset({"first", "second"}))
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "SEARCHES", tmp_path / "searches")
    built = types.SimpleNamespace(daily=(pd.DataFrame({"x": [0.0]}), None))
    monkeypatch.setattr(matrix, "series", lambda: None)
    monkeypatch.setattr(matrix, "matrices", lambda series: built)
    monkeypatch.setattr(matrix, "write", lambda found, folder: None)
    monkeypatch.setattr(record, "fingerprints", lambda: dict(state.fingerprints))
    monkeypatch.setattr(search, "searcher", lambda family: types.SimpleNamespace(MATRIX="daily"))
    monkeypatch.setattr(figures, "draw", lambda where: None)
    # Una sola llavor, sigui quina sigui la variant d'assaig que tria `chosen`.
    real = parameters.chosen
    monkeypatch.setattr(parameters, "chosen", lambda smoke: real(smoke) | {"SEEDS": (0,)})
    state.module = module
    state.main = lambda: run.main(types.SimpleNamespace(smoke=True, family=None))
    return state


def _name(number):
    return f"{paths.SMOKE_PREFIX}{datetime.date.today().isoformat()}_{number:02d}"


def test_a_killed_run_is_continued_each_family_on_the_plan_it_recorded(fake, capsys):
    # La segona família peta la primera vegada, i mentrestant el seu número de lloc canvia:
    # la passada que la continua ha d'ajustar el que diu el seu registre, i no tornar a
    # ajustar la primera, que ja hi és sencera.
    fake.failing = {"second"}
    with pytest.raises(RuntimeError):
        fake.main()
    fake.failing = set()
    fake.module.FAMILIES["second"]["placeholder"] = {"value": 2}
    assert fake.main() == 0
    assert [family for family, _ in fake.calls] == ["first", "second", "second"]
    assert fake.calls[-1][1] == {0: [{"rank": 0, "numbers": {"value": 1},
                                      "unsearched": True}]}
    assert capsys.readouterr().out.count(f"passada {_name(1)}\n") == 2


def test_a_finished_run_is_never_continued(fake, capsys):
    assert fake.main() == 0
    assert fake.main() == 0
    out = capsys.readouterr().out
    assert f"passada {_name(1)}\n" in out and f"passada {_name(2)}\n" in out


def test_a_run_made_by_other_code_is_refused_and_not_continued(fake, capsys):
    fake.failing = {"second"}
    with pytest.raises(RuntimeError):
        fake.main()
    fake.failing = set()
    fake.fingerprints["code"] = "un altre codi"
    calls = len(fake.calls)
    assert fake.main() == 1
    assert len(fake.calls) == calls
    assert _name(1) in capsys.readouterr().out.splitlines()[-1]
