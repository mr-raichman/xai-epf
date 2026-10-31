import datetime
import io
import json
import math
import shutil
import sys
import types

import numpy as np
import optuna
import pandas as pd
import pytest

from epf import Declined, Refusal, information, parameters, paths, terminal
from epf.contract import matrix, record, searches
from epf.parameters import DEVELOPMENT
from epf.run import families, figures, pool, run, tables

# La graella de la cerca falsa, amb més punts que trials: cap trial continuat no en
# torna a fer cap.
GRID = (5, 0, 4, 1, 3, 2)

# L'instant en què comença tota passada nova, i el seu information set.
MINUTE = datetime.datetime(2026, 10, 22, 14, 30)
SET = "WITH_PRICE"


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
    canvia aquí: `WORKERS` alhora, la més antiga primer. La comanda té un terminal, on
    `typed` escriu les respostes. Torna l'estat, amb `main`, que en fa una.
    """
    state = types.SimpleNamespace(
        calls=[],
        failing={},
        diverging=set(),
        opened=0,
        fingerprints={
            "code": "codi",
            "dataset": "conjunt",
            "matrices": "matrius",
            "environment": "llibres",
        },
        values={"SEEDS": (0, 1), "TRIALS": 4, "BEST": 2},
    )

    def raised(family, seed, kind):
        if (family, seed, kind) in state.failing:
            raise state.failing[family, seed, kind]("interrompuda")

    def train(family, folder, seed, configuration, explained_days, built):
        state.calls.append(("fit", family, seed, configuration.get("numbers")))
        raised(family, seed, "fit")
        path = folder / paths.MODEL.format(seed=seed, rank=configuration["rank"])
        np.savez(path, r2=np.float64(0.5), mae=np.float64(1.0))
        return None if family == "third" else "aturat"

    def trial(family, drawn, train, validate, seed):
        order = sum(call[:3] == ("trial", family, seed) for call in state.calls)
        state.calls.append(("trial", family, seed, drawn))
        raised(family, seed, "trial")
        if (family, seed, order) in state.diverging:
            return {"value": drawn}, math.nan, {}
        return {"value": drawn}, float(drawn), {}

    trainer = types.ModuleType("fake_trainer")
    trainer.FAMILIES = {
        "first": {
            "fixed": {},
            "kept": "BEST",
            "searched_as": "first",
            "ljung_box_tested": True,
        },
        "second": {
            "fixed": {},
            "kept": 1,
            "searched_as": "first",
            "ljung_box_tested": True,
        },
        "third": {
            "fixed": {},
            "kept": 0,
            "searched_as": "third",
            "ljung_box_tested": False,
        },
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
                "model": family,
                "train": "fake_trainer",
                "search": "fake_searcher" if family == "first" else None,
                "figures": None,
            }
            for family in trainer.FAMILIES
        },
    )
    monkeypatch.setattr(families, "EVERY", tuple(trainer.FAMILIES))
    monkeypatch.setattr(paths, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(paths, "SMOKE_OUTPUT", tmp_path / "assaig_output")
    clock = types.SimpleNamespace(now=lambda: MINUTE)
    monkeypatch.setattr(run, "datetime", types.SimpleNamespace(datetime=clock))
    days = pd.date_range(DEVELOPMENT[0], periods=14, freq="D")
    table = (
        pd.DataFrame({"x": np.zeros(len(days))}, index=days),
        pd.Series(np.zeros(len(days)), index=days),
    )
    built = types.SimpleNamespace(
        daily=table,
        hourly=table,
        deleted={},
        information=information.resolved(SET),
    )
    monkeypatch.setattr(matrix, "series", lambda: None)
    monkeypatch.setattr(
        matrix,
        "matrices",
        lambda series, information_set: types.SimpleNamespace(
            **vars(built) | {"information": information_set}
        ),
    )
    monkeypatch.setattr(matrix, "write", lambda found, folder: None)
    monkeypatch.setattr(record, "fingerprints", lambda built: dict(state.fingerprints))
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
    monkeypatch.setattr(terminal, "interactive", lambda: True)
    state.typed = lambda text: monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    state.typed("")
    state.module = trainer
    state.searcher = searcher
    state.main = lambda smoke="SMOKE", answer=None, family=None, set_name=SET: run.main(
        types.SimpleNamespace(
            smoke=smoke, family=family, answer=answer, set_name=set_name
        )
    )
    return state


def _name():
    """Torna el nom de la passada d'assaig que comença a `MINUTE`."""
    return paths.next_run(True, SET, MINUTE).name


def _recorded(family, where=None):
    """Torna el registre d'una família a la passada `where`, la de `MINUTE` sense."""
    where = where or paths.next_run(True, SET, MINUTE)
    path = where / family / paths.FAMILY_RECORD
    return json.loads(path.read_text(encoding="utf-8"))


def _ranked(seed):
    """Torna els valors de la cerca de `first` a una seed, del millor al pitjor."""
    path = searches.folder("SMOKE", SET, "first") / searches.RECORD.format(seed=seed)
    found = json.loads(path.read_text(encoding="utf-8"))
    return [one["numbers"]["value"] for one in found["ranked"]]


def test_each_searched_seed_s_study_is_copied_into_its_searcher_s_folder(fake):
    fake.main()
    where = paths.next_run(True, SET, MINUTE)
    for seed in (0, 1):
        name = searches.STUDY.format(seed=seed)
        found = [
            [(one.number, one.state, one.params, one.value) for one in study.trials]
            for study in (
                searches.stored(
                    searches.folder("SMOKE", SET, "first") / name, "first", seed
                ),
                searches.stored(where / "first" / name, "first", seed),
            )
        ]
        assert len(found[1]) == 4 and found[1] == found[0]
    assert not list((where / "second").glob("*.db"))


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


def test_the_units_go_family_by_family_a_seeds_fits_right_after_its_search(fake):
    """Amb dues comandes, cap ajust no començava abans que s'acabés tota la cerca."""
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == (
        [("trial", "first", 0)] * 4
        + [("fit", "first", 0)] * 2
        + [("trial", "first", 1)] * 4
        + [("fit", "first", 1)] * 2
        + [("fit", "second", 0), ("fit", "second", 1)]
        + [("fit", "third", 0), ("fit", "third", 1)]
    )


def test_a_search_with_no_trials_set_searches_its_whole_grid(fake):
    fake.values["TRIALS"] = None
    fake.main()
    for seed in (0, 1):
        assert sorted(_ranked(seed)) == sorted(GRID)


def test_a_seeds_next_wave_is_asked_once_its_last_has_returned_whole(fake, monkeypatch):
    """Demanada abans, l'onada sortiria de quins trials han tornat primer."""
    fake.searcher.TRIALS_PER_WAVE = 2
    monkeypatch.setattr(run, "WORKERS", 3)
    real = searches.asked

    def asked(study, searcher, family):
        seed = int(study.study_name.rsplit("_", 1)[1])
        fake.calls.append(("ask", family, seed, None))
        return real(study, searcher, family)

    monkeypatch.setattr(searches, "asked", asked)
    fake.main()
    assert [kind for kind, _, seed, _ in fake.calls if seed == 0 and kind != "fit"] == [
        "ask",
        "ask",
        "trial",
        "trial",
    ] * 2


def test_the_log_holds_a_line_a_seeds_search_and_a_fit_that_stopped(fake):
    """Un ajust aturat pel sostre no es veia enlloc."""
    fake.main()
    log = paths.runs(True, SET) / _name() / paths.WORKERS_LOG
    assert log.read_text(encoding="utf-8").splitlines() == [
        "first seed 0: cerca acabada amb els seus 4 trials",
        "first seed 0 fit 0: aturat",
        "first seed 0 fit 1: aturat",
        "first seed 1: cerca acabada amb els seus 4 trials",
        "first seed 1 fit 0: aturat",
        "first seed 1 fit 1: aturat",
        "second seed 0 fit 0: aturat",
        "second seed 1 fit 0: aturat",
    ]


def test_a_failed_trial_stops_its_seed_and_a_continuation_replaces_it(fake):
    """Una seed amb un trial de menys no es pot planificar, i continuar l'acaba."""
    fake.failing = {("first", 1, "trial"): RuntimeError}
    with pytest.raises(
        Refusal,
        match=f"^la passada {_name()} no és acabada: 1 seed ha fallat, first seed 1; "
        f"llegir .*/{_name()}/workers.log$",
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
    fake.typed("n\ny\n")
    fake.main()
    assert [(kind, seed) for kind, _, seed, _ in fake.calls] == [("trial", 1)] * 4 + [
        ("fit", 1)
    ] * 3
    store = searches.folder("SMOKE", SET, "first") / searches.STUDY.format(seed=1)
    study = optuna.load_study(study_name="first_seed_1", storage=f"sqlite:///{store}")
    assert [one.state.name for one in study.trials] == ["FAIL"] + ["COMPLETE"] * 4


def test_a_diverged_trial_counts_and_its_seed_ranks_the_others(fake):
    """Un trial que no puntua compta entre els de la seed, sense cap altre en lloc seu.

    Com a la cerca de Marcjasz et al. (2023).
    """
    fake.diverging = {("first", 0, 1)}
    fake.main()
    store = searches.folder("SMOKE", SET, "first") / searches.STUDY.format(seed=0)
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
    log = paths.runs(True, SET) / _name() / paths.WORKERS_LOG
    assert "first seed 0 trial 1 ha divergit: nan\n" in log.read_text(encoding="utf-8")


def test_a_continued_run_asks_no_trial_in_place_of_a_diverged_one(fake):
    """Un trial abandonat es torna a demanar, però un que ha divergit ja compta."""
    fake.diverging = {("first", 0, 1)}
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.calls.clear()
    fake.typed("n\ny\n")
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
        match=f"^la passada {_name()} no és acabada: 1 seed ha fallat, first seed 1; "
        f"llegir .*/{_name()}/workers.log$",
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
    planned = _recorded("first")["configurations"]
    fake.calls.clear()
    fake.typed("y\n")
    fake.main()
    assert [kind for kind, *_ in fake.calls] == ["fit"] * 8
    assert _recorded("first")["configurations"] == planned


def test_a_killed_run_is_continued_each_family_on_the_plan_it_recorded(fake, capsys):
    """Refet en continuar, el pla canviaria els números de les famílies que falten."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada: 1 seed ha fallat"):
        fake.main()
    planned = _recorded("second")["configurations"]
    fake.failing = {}
    fake.calls.clear()
    fake.typed("n\ny\n")
    fake.module.FAMILIES["second"]["kept"] = 2
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == [
        ("fit", "second", 0)
    ]
    assert _recorded("second")["configurations"] == planned
    shown = capsys.readouterr().err
    assert f"Passada {_name()}\n" in shown
    assert f"Continuant passada {_name()}\n" in shown


def test_a_search_made_again_never_changes_a_recorded_plan(fake):
    """Esborrada i feta de nou, una cerca tornaria a fer els plans ja desats."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    planned = {
        family: _recorded(family)["configurations"] for family in ("first", "second")
    }
    shutil.rmtree(paths.searches(True, SET))
    fake.failing = {}
    fake.typed("n\ny\n")
    fake.module.FAMILIES["first"]["kept"] = 1
    fake.main()
    assert {
        family: _recorded(family)["configurations"] for family in ("first", "second")
    } == planned


def test_a_finished_run_kept_with_an_n_is_never_continued(fake, capsys):
    """Una passada acabada que es continués en barrejaria dues."""
    fake.main()
    calls = len(fake.calls)
    fake.typed("n\n")
    with pytest.raises(Declined):
        fake.main()
    assert len(fake.calls) == calls
    assert "Continuar la passada" not in capsys.readouterr().err


def _failed_once(fake):
    """Fa una passada a la qual només falta l'ajust de `second` a la seed 0."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.calls.clear()


def test_a_run_begun_with_other_code_is_asked_and_keeps_each_fits_code(fake, capsys):
    """Una edició, encara que fos un comentari, impedia continuar cap passada."""
    _failed_once(fake)
    fake.fingerprints["code"] = "editat"
    fake.typed("n\ny\n")
    fake.main()
    assert [(kind, family, seed) for kind, family, seed, _ in fake.calls] == [
        ("fit", "second", 0)
    ]
    shown = capsys.readouterr().err
    assert (
        f"{_name()}/, començada amb un altre codi\nfirst   2/2 seeds    4/4 fits\n"
        "second  1/2 seeds    1/2 fits\nthird   2/2 seeds    2/2 fits\n"
    ) in shown
    assert f"Continuar la passada {_name()} amb un altre codi? [y/n] " in shown
    assert _recorded("second")["codes"] == {"0": {"0": "editat"}, "1": {"0": "codi"}}


def test_continue_answers_the_question_on_other_code_without_a_terminal(
    fake, monkeypatch, capsys
):
    _failed_once(fake)
    fake.fingerprints["code"] = "editat"
    monkeypatch.setattr(terminal, "interactive", lambda: False)
    fake.main(answer="continue")
    assert [(kind, family) for kind, family, *_ in fake.calls] == [("fit", "second")]
    assert (
        f"Continuar la passada {_name()} amb un altre codi? [y/n] y (--continue)\n"
        in capsys.readouterr().err
    )


def test_a_run_recorded_without_each_fits_code_is_continued(fake):
    """Cap passada d'abans que es desés el codi de cada ajust no en té cap."""
    _failed_once(fake)
    for family in ("first", "second", "third"):
        path = paths.next_run(True, SET, MINUTE) / family / paths.FAMILY_RECORD
        found = json.loads(path.read_text(encoding="utf-8"))
        del found["codes"]
        paths.write_record(path, found)
    fake.typed("n\ny\n")
    fake.main()
    assert _recorded("second")["codes"] == {"0": {"0": "codi"}}


def test_a_smoke_run_is_never_continued_under_another_variant(fake, capsys):
    """Triar la variant editant `parameters.py` canviava el codi, que ho refusava."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.typed("n\n")
    calls = len(fake.calls)
    with pytest.raises(Declined):
        fake.main("SMOKE_1")
    assert len(fake.calls) == calls
    shown = capsys.readouterr().err
    assert f'La passada {_name()} es va començar amb "--smoke"\n' in shown


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


def test_each_fit_records_its_start_and_load_beside_its_seconds(fake, monkeypatch):
    monkeypatch.setattr(record, "machine_load", lambda: (1.0, 2.0, 3.0))
    fake.main()
    started = MINUTE.astimezone().isoformat(timespec="seconds")
    for family in ("first", "second", "third"):
        found = _recorded(family)
        for field, value in (("started", started), ("load", [1.0, 2.0, 3.0])):
            assert found[field] == {
                seed: {rank: value for rank in ranks}
                for seed, ranks in found["seconds"].items()
            }


def test_a_family_reading_a_search_none_makes_ends_in_an_error(fake, monkeypatch):
    """Sense pla ni cap seed fallada, la passada en dibuixava les figures i acabava."""
    drawn = []
    monkeypatch.setattr(figures, "draw", drawn.append)
    fake.module.FAMILIES["second"]["searched_as"] = "absent"
    with pytest.raises(RuntimeError, match="1 família no és completa, second$"):
        fake.main()
    assert not drawn
    assert (paths.next_run(True, SET, MINUTE) / tables.SCORES).exists()


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


def test_the_question_is_asked_again_until_y_or_n_in_either_case(fake, capsys):
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    fake.typed("x\nN\nY\n")
    fake.main()
    shown = capsys.readouterr().err
    assert shown.count("Substituir el que hi ha en disc? [y/n] ") == 2
    assert shown.count(f"Continuar la passada {_name()}? [y/n] ") == 1
    assert (
        f"{_name()}/\nfirst   2/2 seeds    4/4 fits\nsecond  1/2 seeds    1/2 fits\n"
        "third   2/2 seeds    2/2 fits\n"
    ) in shown
    assert f"Continuant passada {_name()}\n" in shown


def test_a_study_made_otherwise_is_listed_and_goes_with_a_y(fake, capsys):
    fake.main()
    fake.searcher.FIXED = {"patience": 2}
    fake.calls.clear()
    fake.typed("y\n")
    fake.main()
    shown = capsys.readouterr().err
    for seed in (0, 1):
        assert (
            f"seed_{seed}.db: 4 trials, es va fer amb uns altres números fixos\n"
            in shown
        )
    assert [kind for kind, *_ in fake.calls].count("trial") == 8


@pytest.mark.parametrize(
    "terminal_, typed, why",
    [
        (False, "", "sense terminal, no es pot preguntar"),
        (True, "", "sense resposta, es conserva el que hi ha en disc"),
    ],
    ids=["sense terminal", "sense resposta"],
)
def test_without_a_terminal_or_an_answer_nothing_goes(
    fake, monkeypatch, terminal_, typed, why
):
    fake.main()
    monkeypatch.setattr(terminal, "interactive", lambda: terminal_)
    fake.typed(typed)
    with pytest.raises(Refusal, match=f"^{why}"):
        fake.main()
    assert paths.named_runs(True, SET) == [paths.next_run(True, SET, MINUTE)]


def test_an_n_with_no_run_to_continue_stops_and_keeps_what_is_on_disk(fake):
    fake.main()
    first = paths.next_run(True, SET, MINUTE)
    kept = first.rename(first.with_name("kept"))
    fake.typed("n\n")
    with pytest.raises(Declined):
        fake.main()
    assert kept.exists() and not paths.named_runs(True, SET)


@pytest.mark.parametrize("code", ["codi", "editat"], ids=["el codi", "un altre codi"])
def test_an_n_to_continuing_stops_and_fits_nothing(fake, code):
    _failed_once(fake)
    fake.fingerprints["code"] = code
    fake.typed("n\nn\n")
    with pytest.raises(Declined):
        fake.main()
    assert not fake.calls


def test_the_flags_answer_both_questions_without_a_terminal(fake, monkeypatch, capsys):
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    monkeypatch.setattr(terminal, "interactive", lambda: False)
    fake.calls.clear()
    fake.main(answer="continue")
    assert [(kind, family) for kind, family, *_ in fake.calls] == [("fit", "second")]
    with pytest.raises(Declined):
        fake.main(answer="continue")
    fake.calls.clear()
    fake.main(answer="replace")
    assert [kind for kind, *_ in fake.calls] == ["fit"] * 8
    shown = capsys.readouterr().err
    assert "Substituir el que hi ha en disc? [y/n] n (--continue)\n" in shown
    assert f"Continuar la passada {_name()}? [y/n] y (--continue)\n" in shown
    assert "Substituir el que hi ha en disc? [y/n] y (--replace)\n" in shown


def test_a_run_whose_named_families_are_complete_is_not_offered_to_continue(
    fake, capsys
):
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    fake.failing = {}
    calls = len(fake.calls)
    fake.typed("n\n")
    with pytest.raises(Declined):
        fake.main(family=["third"])
    assert len(fake.calls) == calls
    assert "Continuar la passada" not in capsys.readouterr().err


def test_a_run_with_a_failed_seed_leaves_the_others_scores(fake):
    """Escrita només en dibuixar, una seed fallada deixava la passada sense la taula."""
    fake.failing = {("second", 0, "fit"): RuntimeError}
    with pytest.raises(Refusal, match="no és acabada"):
        fake.main()
    path = paths.next_run(True, SET, MINUTE) / tables.SCORES
    rows = [row.split(",")[:3] for row in path.read_text(encoding="utf-8").splitlines()]
    assert rows[1:] == [
        ["first", "0", "0"],
        ["first", "0", "1"],
        ["first", "1", "0"],
        ["first", "1", "1"],
        ["second", "1", "0"],
        ["third", "0", "0"],
        ["third", "1", "0"],
    ]


def test_a_set_s_run_replaces_its_own_run_and_studies_alone(fake):
    """Una passada d'un information set substituïa la de qualsevol altre."""
    fake.main()
    for _ in range(2):
        fake.main(set_name="WITHOUT_PRICE", answer="replace")
    for set_name in (SET, "WITHOUT_PRICE"):
        assert paths.named_runs(True, set_name) == [
            paths.next_run(True, set_name, MINUTE)
        ]
        assert (searches.folder("SMOKE", set_name, "first") / "seed_0.db").exists()


def test_a_family_whose_columns_the_set_lacks_is_left_out_and_said():
    """El preu sol no té les previsions, que el sòl llegeix pel seu nom."""
    rows = pd.DataFrame(columns=["price_lag1_h00", "hour"])
    built = types.SimpleNamespace(hourly=(rows, None))
    kept, said = run._readable(["floor", "naive_previous_day"], built)
    assert kept == ["naive_previous_day"]
    assert said == ["floor: no s'ajusta, sense demand_fcst, wind_fcst, solar_fcst"]
