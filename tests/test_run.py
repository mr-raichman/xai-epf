import datetime
import json
import math
import shutil
import sys
import types

import numpy as np
import optuna
import pandas as pd
import pytest

from epf import Refusal, parameters, paths
from epf.contract import matrix, record, searches
from epf.parameters import DEVELOPMENT
from epf.run import families, figures, pool, run

# La graella de la cerca falsa, amb més punts que trials: cap trial continuat no en
# torna a fer cap.
GRID = (5, 0, 4, 1, 3, 2)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Prepara passades d'assaig d'un model fals de tres famílies, que no ajusta res.

    `first` es cerca i guarda les `BEST` millors configuracions de cada seed, `second`
    llegeix la cerca de `first` i en guarda la millor, com la sonda de datació, i
    `third` no es cerca. La cerca falsa proposa a cada trial un punt de `GRID` i el
    puntua pel seu valor, en studies d'Optuna de debò, o amb un NaN el trial de
    `diverging`, `(família, seed, ordre)`. Cada ajust desa un fitxer buit, i un trial o
    un ajust de `failing`, `(família, seed, tipus)`, hi llança la seva excepció. Les
    tasques es fan aquí mateix, i no en processos nous, que no veurien res del que es
    canvia aquí: `WORKERS` alhora, la més antiga primer. Torna l'estat, amb `main`, que
    en fa una.
    """
    state = types.SimpleNamespace(
        calls=[],
        failing={},
        diverging=set(),
        opened=0,
        fingerprints={"code": "codi", "dataset": "conjunt", "environment": "llibres"},
        values={"SEEDS": (0, 1), "TRIALS": 4, "BEST": 2},
    )

    def raised(family, seed, kind):
        if (family, seed, kind) in state.failing:
            raise state.failing[family, seed, kind]("interrompuda")

    def train(family, folder, seed, configuration, explained_days):
        state.calls.append(("fit", family, seed, configuration.get("numbers")))
        raised(family, seed, "fit")
        (folder / paths.SEED.format(seed=seed, rank=configuration["rank"])).touch()

    def trial(family, drawn, train, validate, seed):
        order = sum(call[:3] == ("trial", family, seed) for call in state.calls)
        state.calls.append(("trial", family, seed, drawn))
        raised(family, seed, "trial")
        if (family, seed, order) in state.diverging:
            return {"value": drawn}, math.nan
        return {"value": drawn}, float(drawn)

    trainer = types.ModuleType("fake_trainer")
    trainer.FAMILIES = {
        "first": {"fixed": {}, "kept": "BEST", "searched_as": "first"},
        "second": {"fixed": {}, "kept": 1, "searched_as": "first"},
        "third": {"fixed": {}, "kept": 0, "searched_as": "third"},
    }
    trainer.train = train
    searcher = types.ModuleType("fake_searcher")
    searcher.MATRIX = "daily"
    searcher.GRID = {"value": GRID}
    searcher.SPACE = searcher.GRID
    searcher.FIXED = {"patience": 1}
    searcher.drawn = lambda family, one: one.suggest_categorical("value", GRID)
    searcher.trial = trial
    monkeypatch.setitem(sys.modules, "fake_trainer", trainer)
    monkeypatch.setitem(sys.modules, "fake_searcher", searcher)
    monkeypatch.setattr(
        families,
        "MODULES",
        {
            family: {
                "models": (family,),
                "train": "fake_trainer",
                "search": "fake_searcher" if family == "first" else None,
                "figures": None,
            }
            for family in trainer.FAMILIES
        },
    )
    monkeypatch.setattr(families, "EVERY", tuple(trainer.FAMILIES))
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "SEARCHES", tmp_path / "searches")
    days = pd.date_range(DEVELOPMENT[0], periods=14, freq="D")
    built = types.SimpleNamespace(
        daily=(
            pd.DataFrame({"x": np.zeros(len(days))}, index=days),
            pd.Series(np.zeros(len(days)), index=days),
        )
    )
    monkeypatch.setattr(matrix, "series", lambda: None)
    monkeypatch.setattr(matrix, "matrices", lambda series: built)
    monkeypatch.setattr(matrix, "write", lambda found, folder: None)
    monkeypatch.setattr(
        matrix, "refuse_missing_days", lambda found, period, deleted: None
    )
    monkeypatch.setattr(record, "fingerprints", lambda: dict(state.fingerprints))
    monkeypatch.setattr(figures, "draw", lambda where: None)
    opened = searches.opened

    def counted(*arguments):
        state.opened += 1
        return opened(*arguments)

    monkeypatch.setattr(searches, "opened", counted)

    def done(tasks, workers, threads, log, block, on_return=None, on_failure=None):
        tasks, running = iter(tasks), []

        def fed():
            while len(running) < workers:
                task = next(tasks, None)
                if task is None:
                    return
                running.append(task)
                block.start(task[0][0])

        fed()
        while running:
            label, function, arguments = running.pop(0)
            family, seed, kind, _ = label
            try:
                result = function(*arguments)
            except (RuntimeError, Refusal) as error:
                if on_failure is not None:
                    on_failure(label)
                block.fail(family, seed, str(error))
            else:
                if on_return is not None:
                    on_return(label, result)
                block.advance(family, seed, kind)
            fed()
        block.close()

    monkeypatch.setattr(pool, "done", done)
    monkeypatch.setattr(run, "WORKERS", 1)
    real = parameters.chosen
    monkeypatch.setattr(
        parameters, "chosen", lambda smoke, read: real(smoke, read) | state.values
    )
    state.module = trainer
    state.main = lambda smoke="SMOKE": run.main(
        types.SimpleNamespace(smoke=smoke, family=None)
    )
    return state


def _name(number):
    """Torna el nom de la passada d'assaig d'avui que porta el número `number`."""
    return f"{paths.SMOKE_PREFIX}{datetime.date.today().isoformat()}_{number:02d}"


def _recorded(family, number=1):
    """Torna el registre d'una família a la passada d'assaig d'avui `number`."""
    path = paths.RUNS / _name(number) / family / paths.SEED_RECORD
    return json.loads(path.read_text(encoding="utf-8"))


def _ranked(seed):
    """Torna els valors de la cerca de `first` a una seed, del millor al pitjor."""
    path = searches.folder("SMOKE", "first") / searches.RECORD.format(seed=seed)
    found = json.loads(path.read_text(encoding="utf-8"))
    return [one["numbers"]["value"] for one in found["ranked"]]


def test_each_seed_fits_the_best_of_its_own_search(fake):
    """Una seed que ajustés el millor d'una altra en descriuria una altra cerca."""
    fake.main()
    for seed in (0, 1):
        best = _ranked(seed)
        planned = {
            family: [
                one.get("numbers")
                for one in _recorded(family)["configurations"][str(seed)]
            ]
            for family in ("first", "second", "third")
        }
        assert planned == {
            "first": [{"value": value} for value in best[:2]],
            "second": [{"value": best[0]}],
            "third": [None],
        }
    assert _ranked(0)[:2] != _ranked(1)[:2]


def test_a_seeds_fits_come_as_soon_as_its_search_has_every_trial(fake):
    """Amb dues comandes, cap ajust no començava abans que s'acabés tota la cerca."""
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == (
        [("fit", "third", 0), ("fit", "third", 1)]
        + [("trial", "first", 0)] * 4
        + [("fit", "first", 0), ("fit", "first", 0), ("fit", "second", 0)]
        + [("trial", "first", 1)] * 4
        + [("fit", "first", 1), ("fit", "first", 1), ("fit", "second", 1)]
    )


def test_a_failed_trial_stops_its_seed_and_a_continuation_replaces_it(fake):
    """Una seed amb un trial de menys no es pot planificar, i continuar l'acaba."""
    fake.failing = {("first", 1, "trial"): RuntimeError}
    with pytest.raises(
        Refusal,
        match=f"^la passada {_name(1)} no és acabada: 1 seed ha fallat, first seed 1$",
    ):
        fake.main()
    assert [seed for kind, _, seed, _ in fake.calls if kind == "trial"] == [0] * 4 + [1]
    assert {
        (family, seed) for kind, family, seed, _ in fake.calls if kind == "fit"
    } == {
        ("third", 0),
        ("third", 1),
        ("first", 0),
        ("second", 0),
    }
    assert "1" not in _recorded("second")["configurations"]
    fake.failing = {}
    fake.calls.clear()
    fake.main()
    assert [(kind, seed) for kind, _, seed, _ in fake.calls] == [("trial", 1)] * 4 + [
        ("fit", 1)
    ] * 3
    store = searches.folder("SMOKE", "first") / searches.STUDY.format(seed=1)
    study = optuna.load_study(study_name="first_seed_1", storage=f"sqlite:///{store}")
    assert [one.state.name for one in study.trials] == ["FAIL"] + ["COMPLETE"] * 4


def test_a_diverged_trial_counts_and_its_seed_ranks_the_others(fake):
    """Un trial que no puntua compta entre els de la seed, sense cap altre en lloc seu.

    Com a la cerca de Marcjasz et al. (2023).
    """
    fake.diverging = {("first", 0, 1)}
    fake.main()
    store = searches.folder("SMOKE", "first") / searches.STUDY.format(seed=0)
    study = optuna.load_study(study_name="first_seed_0", storage=f"sqlite:///{store}")
    assert [one.state.name for one in study.trials] == [
        "COMPLETE",
        "FAIL",
        "COMPLETE",
        "COMPLETE",
    ]
    assert study.trials[1].user_attrs["diverged"]
    found = json.loads(store.with_suffix(".json").read_text(encoding="utf-8"))
    assert found["trials"] == 4 and len(found["ranked"]) == 3
    assert len(_recorded("first")["configurations"]["0"]) == 2
    log = paths.RUNS / _name(1) / paths.WORKERS_LOG
    assert log.read_text(encoding="utf-8") == "first seed 0 trial 1 ha divergit: nan\n"


def test_a_continued_run_asks_no_trial_in_place_of_a_diverged_one(fake):
    """Un trial abandonat es torna a demanar, però un que ha divergit ja compta."""
    fake.diverging = {("first", 0, 1)}
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.calls.clear()
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == [
        ("fit", "second", 0)
    ]


def test_a_seed_whose_every_trial_diverged_fails_and_the_others_finish(fake, capsys):
    """Sense cap trial que puntuï, la seed no té res a ajustar, i ho ha de dir."""
    fake.values["TRIALS"] = 1
    fake.diverging = {("first", 1, 0)}
    with pytest.raises(
        Refusal,
        match=f"^la passada {_name(1)} no és acabada: 1 seed ha fallat, first seed 1$",
    ):
        fake.main()
    shown = capsys.readouterr().err
    assert "error: first seed 1 ha fallat: el seu trial ha divergit\n" in shown
    assert {
        (family, seed) for kind, family, seed, _ in fake.calls if kind == "fit"
    } == {
        ("third", 0),
        ("third", 1),
        ("first", 0),
        ("second", 0),
    }
    assert set(_recorded("second")["configurations"]) == {"0"}


def test_a_run_after_a_finished_one_asks_no_trial_and_fits_again(fake):
    """Les cerques acabades no es tornen a fer, i la passada nova se'n planifica."""
    fake.main()
    fake.calls.clear()
    fake.main()
    assert [kind for kind, *_ in fake.calls] == ["fit"] * 8
    assert (
        _recorded("first", 2)["configurations"]
        == _recorded("first", 1)["configurations"]
    )


def test_a_killed_run_is_continued_each_family_on_the_plan_it_recorded(fake, capsys):
    """Refet en continuar, el pla canviaria els números de les famílies que falten."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada: 1 seed ha fallat"):
        fake.main()
    planned = _recorded("second")["configurations"]
    fake.failing = {}
    fake.calls.clear()
    fake.module.FAMILIES["second"]["kept"] = 2
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == [
        ("fit", "second", 0)
    ]
    assert _recorded("second")["configurations"] == planned
    shown = capsys.readouterr().err
    assert f"Passada {_name(1)}\n" in shown
    assert f"Passada {_name(1)}, continuada\n" in shown


def test_a_search_made_again_never_changes_a_recorded_plan(fake):
    """Esborrada i feta de nou, una cerca tornaria a fer els plans ja desats."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    planned = {
        family: _recorded(family)["configurations"] for family in ("first", "second")
    }
    shutil.rmtree(paths.SEARCHES)
    fake.failing = {}
    fake.module.FAMILIES["first"]["kept"] = 1
    fake.main()
    assert {
        family: _recorded(family)["configurations"] for family in ("first", "second")
    } == planned


def test_a_finished_run_is_never_continued(fake, capsys):
    """Una passada acabada que es continués en barrejaria dues."""
    fake.main()
    fake.main()
    shown = capsys.readouterr().err
    assert f"Passada {_name(1)}\n" in shown and f"Passada {_name(2)}\n" in shown


def test_a_run_made_by_other_code_is_refused_before_any_study_is_opened(fake):
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.fingerprints["code"] = "un altre codi"
    calls, opened = len(fake.calls), fake.opened
    with pytest.raises(Refusal, match=f"la passada {_name(1)} .* un altre codi;"):
        fake.main()
    assert (len(fake.calls), fake.opened) == (calls, opened)


def test_a_smoke_run_is_never_continued_under_another_variant(fake):
    """Triar la variant editant `parameters.py` canviava el codi, que ho refusava."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    calls = len(fake.calls)
    with pytest.raises(Refusal, match='començar amb "--smoke"; continuar amb "uv run'):
        fake.main("SMOKE_1")
    assert len(fake.calls) == calls


def test_each_fits_seconds_are_kept_as_soon_as_it_returns(fake):
    """Escrits en acabar la passada, un Ctrl+C perdia els segons de tots els ajustos."""
    fake.failing = {("second", 1, "fit"): KeyboardInterrupt}
    with pytest.raises(KeyboardInterrupt):
        fake.main()
    seconds = _recorded("first")["seconds"]
    assert {seed: sorted(ranks) for seed, ranks in seconds.items()} == {
        "0": ["0", "1"],
        "1": ["0", "1"],
    }


def test_a_family_reading_a_search_none_makes_ends_in_an_error(fake, monkeypatch):
    """Sense pla ni cap seed fallada, la passada en dibuixava les figures i acabava."""
    drawn = []
    monkeypatch.setattr(figures, "draw", drawn.append)
    fake.module.FAMILIES["second"]["searched_as"] = "absent"
    with pytest.raises(RuntimeError, match="1 família no és completa, second$"):
        fake.main()
    assert not drawn


def test_a_fit_that_saves_no_file_is_not_taken_for_done(fake, monkeypatch):
    monkeypatch.setattr(fake.module, "train", lambda *arguments: None)
    with pytest.raises(RuntimeError, match="3 famílies no són completes, la primera"):
        fake.main()


def test_a_task_after_an_edit_fails_its_seed(fake, monkeypatch, capsys):
    """Una edició a mitja passada entrava a les tasques següents sense dir-ho."""
    real = fake.module.train

    def edited(*arguments):
        real(*arguments)
        fake.fingerprints["code"] = "editat"

    monkeypatch.setattr(fake.module, "train", edited)
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    shown = capsys.readouterr().err
    assert "ha fallat: la passada es va començar amb un altre codi" in shown
    assert len([call for call in fake.calls if call[0] == "fit"]) == 1
