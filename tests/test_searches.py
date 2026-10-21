import datetime
import types

import numpy as np
import pandas as pd
import pytest

from epf import paths
from epf.contract import split as split_module
from epf.contract.searches import (
    asked,
    completed,
    configurations,
    folder,
    opened,
    record,
    save,
    searched,
    told,
    tried,
)
from epf.contract.split import split

# Un període curt de setmanes senceres, de dimarts a dilluns, amb una setmana suprimida.
PERIOD = (datetime.date(2019, 1, 1), datetime.date(2019, 7, 29))
DELETED = (datetime.date(2019, 4, 30), datetime.date(2019, 5, 6))

# Les empremtes que desa cada trial.
FINGERPRINTS = {"code": "codi", "dataset": "conjunt", "environment": "biblioteques"}


def _table():
    """Torna `(X, y)` de `PERIOD`: una columna que compta els dies, i un preu zero."""
    days = pd.date_range(*PERIOD, freq="D")
    return (
        pd.DataFrame({"x": np.arange(len(days), dtype=float)}, index=days),
        pd.Series(np.zeros(len(days)), index=days),
    )


def _searcher(grid, seen):
    """Torna un cercador com el `search.py` d'un model, que no ajusta res.

    Desa a `seen` els dies que veu cada trial, i la puntuació és el valor proposat.
    """

    def drawn(family, optuna_trial):
        if grid is None:
            return optuna_trial.suggest_float("value", 0.0, 1.0)
        return optuna_trial.suggest_categorical("value", grid["value"])

    def trial(family, drawn, train, validate, seed):
        seen.append((train[0].index, validate[0].index))
        return {"value": drawn}, float(drawn)

    return types.SimpleNamespace(
        MATRIX="daily",
        GRID=grid,
        SPACE=grid or {"value": (0.0, 1.0)},
        FIXED={"patience": 1},
        drawn=drawn,
        trial=trial,
    )


def _study(store, searcher, X, y, parts, seed, trials):
    """Fa en aquest procés els trials que falten fins a `trials`, com una cerca."""
    study = opened(store, "fake", searcher, seed, X.columns, FINGERPRINTS["dataset"])
    while len(completed(study)) < trials:
        optuna_trial, drawn = asked(study, searcher, "fake")
        numbers, score = tried(searcher, "fake", drawn, X, y, parts, seed)
        told(study, optuna_trial, numbers, score, FINGERPRINTS)
    return study


def test_a_trial_sees_the_training_and_validation_days_and_no_test_day():
    """Un trial que veiés la prova en triaria els números."""
    X, y = _table()
    parts = split(0, PERIOD, DELETED)
    seen = []
    tried(_searcher({"value": tuple(range(8))}, seen), "fake", 3, X, y, parts, 0)
    ((train, validate),) = seen
    assert train.equals(parts.training) and validate.equals(parts.validation)
    assert not train.isin(parts.test).any() and not validate.isin(parts.test).any()


@pytest.mark.parametrize(
    "grid", [{"value": tuple(range(8))}, None], ids=["graella", "continu"]
)
def test_a_continued_search_does_not_repeat_its_trials(tmp_path, grid):
    X, y = _table()
    parts = split(0, PERIOD, DELETED)
    searcher = _searcher(grid, [])
    _study(tmp_path / "study.db", searcher, X, y, parts, 0, 3)
    study = _study(tmp_path / "study.db", searcher, X, y, parts, 0, 6)
    values = [one.params["value"] for one in completed(study)]
    assert len(values) == 6 and len(set(values)) == 6


def _saved(tmp_path, monkeypatch, X, y):
    """Fa una cerca de dos trials de `fake` a la seed 0, i la desa a l'assaig.

    Torna el seu registre.
    """
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    where = folder(True, "fake")
    where.mkdir(parents=True)
    study = _study(
        where / "seed_0.db",
        _searcher({"value": (3, 1, 2)}, []),
        X,
        y,
        split(0, PERIOD, DELETED),
        0,
        2,
    )
    found = record(study, "fake", 0, True)
    save(found, True)
    return found


def test_a_saved_search_is_read_for_its_own_family_and_seed_alone(
    tmp_path, monkeypatch
):
    """Llegit per una altra seed, un registre donaria números que no són seus."""
    X, y = _table()
    found = _saved(tmp_path, monkeypatch, X, y)
    assert (
        searched(
            True,
            "fake",
            0,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
        == found
    )
    assert [one["score"] for one in found["ranked"]] == sorted(
        one["score"] for one in found["ranked"]
    )
    assert (
        searched(
            True,
            "fake",
            1,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
        is None
    )
    assert (
        searched(
            True,
            "other",
            0,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
        is None
    )
    assert (
        searched(
            False,
            "fake",
            0,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
        is None
    )


def test_a_search_made_on_other_columns_is_neither_read_nor_continued(
    tmp_path, monkeypatch
):
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    other = X.assign(z=0.0)
    with pytest.raises(ValueError):
        searched(
            True,
            "fake",
            0,
            other.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
    with pytest.raises(ValueError):
        opened(
            folder(True, "fake") / "seed_0.db",
            "fake",
            _searcher({"value": (3, 1, 2)}, []),
            0,
            other.columns,
            FINGERPRINTS["dataset"],
        )


def test_a_search_made_with_another_PROJECT_SEED_is_neither_read_nor_continued(
    tmp_path, monkeypatch
):
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    monkeypatch.setattr("epf.contract.searches.PROJECT_SEED", 1)
    with pytest.raises(ValueError):
        searched(
            True,
            "fake",
            0,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
    with pytest.raises(ValueError):
        opened(
            folder(True, "fake") / "seed_0.db",
            "fake",
            _searcher({"value": (3, 1, 2)}, []),
            0,
            X.columns,
            FINGERPRINTS["dataset"],
        )


def test_a_search_made_on_another_deal_is_neither_read_nor_continued(
    tmp_path, monkeypatch
):
    """Les setmanes d'una seed canvien amb una altra derivació o una altra NumPy."""
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    real = split_module.seeded
    # Una altra partició de la mateixa seed, i la mateixa `PROJECT_SEED`.
    monkeypatch.setattr(
        split_module,
        "seeded",
        lambda label, *parts: real(label, *parts, *([1] if label == "split" else [])),
    )
    with pytest.raises(ValueError, match="sobre altres files"):
        searched(
            True,
            "fake",
            0,
            X.columns,
            _searcher({"value": (3, 1, 2)}, []),
            FINGERPRINTS["dataset"],
        )
    with pytest.raises(ValueError, match="sobre altres files"):
        opened(
            folder(True, "fake") / "seed_0.db",
            "fake",
            _searcher({"value": (3, 1, 2)}, []),
            0,
            X.columns,
            FINGERPRINTS["dataset"],
        )


def test_a_search_made_on_another_dataset_is_neither_read_nor_continued(
    tmp_path, monkeypatch
):
    """Uns trials puntuats sobre unes altres dades es compararien amb els d'ara."""
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    with pytest.raises(ValueError, match="amb un altre dataset"):
        searched(
            True, "fake", 0, X.columns, _searcher({"value": (3, 1, 2)}, []), "un altre"
        )
    with pytest.raises(ValueError, match="amb un altre dataset"):
        opened(
            folder(True, "fake") / "seed_0.db",
            "fake",
            _searcher({"value": (3, 1, 2)}, []),
            0,
            X.columns,
            "un altre",
        )


def test_a_search_made_on_another_space_or_fixed_numbers_is_neither_read_nor_continued(
    tmp_path, monkeypatch
):
    """Uns trials d'un altre espai o d'altres números no es comparen amb els d'ara."""
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    for change, why in (
        ({"SPACE": {"value": (3, 1, 2, 4)}}, "sobre un altre espai"),
        ({"FIXED": {"patience": 2}}, "amb uns altres números fixos"),
    ):
        changed = types.SimpleNamespace(
            **vars(_searcher({"value": (3, 1, 2)}, [])) | change
        )
        with pytest.raises(ValueError, match=why):
            searched(True, "fake", 0, X.columns, changed, FINGERPRINTS["dataset"])
        with pytest.raises(ValueError, match=why):
            opened(
                folder(True, "fake") / "seed_0.db",
                "fake",
                changed,
                0,
                X.columns,
                FINGERPRINTS["dataset"],
            )


# Una família que es cerca i se n'ajusten les tres millors configuracions de cada seed.
FAKE = {"fixed": {}, "placeholder": {"value": 0}, "kept": 3, "searched_as": "fake"}


def _searched(tmp_path, monkeypatch, seeds):
    """Fa una cerca de quatre trials de `fake` a cada seed de `seeds`, a l'assaig.

    Torna les columnes sobre les quals es van fer.
    """
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    X, y = _table()
    where = folder(True, "fake")
    where.mkdir(parents=True)
    for seed in seeds:
        study = _study(
            where / f"seed_{seed}.db",
            _searcher({"value": (4, 1, 3, 2)}, []),
            X,
            y,
            split(seed, PERIOD, DELETED),
            seed,
            4,
        )
        save(record(study, "fake", seed, True), True)
    return X.columns


def test_with_no_seed_searched_each_seed_fits_the_placeholders_marked(
    tmp_path, monkeypatch
):
    columns = _searched(tmp_path, monkeypatch, [])
    found = configurations(
        True,
        FAKE,
        columns,
        _searcher({"value": (4, 1, 3, 2)}, []),
        (0, 1),
        10,
        FINGERPRINTS["dataset"],
    )
    assert found == {
        seed: [{"rank": 0, "numbers": {"value": 0}, "unsearched": True}]
        for seed in (0, 1)
    }


def test_with_every_seed_searched_each_fits_its_best_and_the_date_probe_the_first(
    tmp_path, monkeypatch
):
    columns = _searched(tmp_path, monkeypatch, (0, 1))
    found = configurations(
        True,
        FAKE,
        columns,
        _searcher({"value": (4, 1, 3, 2)}, []),
        (0, 1),
        10,
        FINGERPRINTS["dataset"],
    )
    for seed in (0, 1):
        assert [one["rank"] for one in found[seed]] == [0, 1, 2]
        assert [one["numbers"]["value"] for one in found[seed]] == [1, 2, 3]
        assert all(one["trials"] == 4 for one in found[seed])
    # La sonda de datació en guarda una, la millor, de la cerca d'una altra família: amb
    # més, no seria la sonda que diu.
    probe = {"fixed": {}, "placeholder": None, "kept": 1, "searched_as": "fake"}
    alone = configurations(
        True,
        probe,
        columns,
        _searcher({"value": (4, 1, 3, 2)}, []),
        (0, 1),
        10,
        FINGERPRINTS["dataset"],
    )
    assert {
        seed: [one["numbers"]["value"] for one in ones] for seed, ones in alone.items()
    } == {0: [1], 1: [1]}


def test_the_tree_keeps_the_commands_BEST(tmp_path, monkeypatch):
    """Un assaig amb `BEST` a 1 ajustava les deu millors, el `BEST` de la mesura."""
    columns = _searched(tmp_path, monkeypatch, (0, 1))
    found = configurations(
        True,
        FAKE | {"kept": "BEST"},
        columns,
        _searcher({"value": (4, 1, 3, 2)}, []),
        (0, 1),
        2,
        FINGERPRINTS["dataset"],
    )
    assert {
        seed: [one["numbers"]["value"] for one in ones] for seed, ones in found.items()
    } == {0: [1, 2], 1: [1, 2]}


def test_a_search_of_some_seeds_only_is_refused(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    with pytest.raises(ValueError):
        configurations(
            True,
            FAKE,
            columns,
            _searcher({"value": (4, 1, 3, 2)}, []),
            (0, 1),
            10,
            FINGERPRINTS["dataset"],
        )


def test_a_family_never_searched_fits_once_with_no_numbers(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    never = FAKE | {"placeholder": None, "kept": 0}
    assert configurations(
        True,
        never,
        columns,
        _searcher({"value": (4, 1, 3, 2)}, []),
        (0, 1),
        10,
        FINGERPRINTS["dataset"],
    ) == {
        0: [{"rank": 0}],
        1: [{"rank": 0}],
    }
