import datetime
import math
import types

import numpy as np
import pandas as pd
import pytest

from epf import Refusal, paths
from epf.contract import split as split_module
from epf.contract.searches import (
    asked,
    completed,
    configurations,
    folder,
    opened,
    record,
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


def _studied(tmp_path, monkeypatch, X, y):
    """Torna el study d'una cerca de dos trials de `fake` a la seed 0, a l'assaig."""
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    where = folder(True, "fake")
    where.mkdir(parents=True)
    return _study(
        where / "seed_0.db",
        _searcher({"value": (3, 1, 2)}, []),
        X,
        y,
        split(0, PERIOD, DELETED),
        0,
        2,
    )


def _continued(searcher, columns, dataset=FINGERPRINTS["dataset"]):
    """Torna a obrir el study de `fake` a la seed 0 amb el que es cerca ara."""
    return opened(
        folder(True, "fake") / "seed_0.db", "fake", searcher, 0, columns, dataset
    )


def test_a_search_made_on_other_columns_is_not_continued(tmp_path, monkeypatch):
    X, y = _table()
    _studied(tmp_path, monkeypatch, X, y)
    with pytest.raises(Refusal, match="sobre altres files"):
        _continued(_searcher({"value": (3, 1, 2)}, []), X.assign(z=0.0).columns)


def test_a_search_made_with_another_PROJECT_SEED_is_not_continued(
    tmp_path, monkeypatch
):
    X, y = _table()
    _studied(tmp_path, monkeypatch, X, y)
    monkeypatch.setattr("epf.contract.searches.PROJECT_SEED", 1)
    with pytest.raises(Refusal, match="amb una altra PROJECT_SEED"):
        _continued(_searcher({"value": (3, 1, 2)}, []), X.columns)


def test_a_search_made_on_another_deal_is_not_continued(tmp_path, monkeypatch):
    """Les setmanes d'una seed canvien amb una altra derivació o una altra NumPy."""
    X, y = _table()
    _studied(tmp_path, monkeypatch, X, y)
    real = split_module.seeded
    # Una altra partició de la mateixa seed, i la mateixa `PROJECT_SEED`.
    monkeypatch.setattr(
        split_module,
        "seeded",
        lambda label, *parts: real(label, *parts, *([1] if label == "split" else [])),
    )
    with pytest.raises(Refusal, match="sobre altres files"):
        _continued(_searcher({"value": (3, 1, 2)}, []), X.columns)


def test_a_search_made_on_another_dataset_is_not_continued(tmp_path, monkeypatch):
    """Uns trials puntuats sobre unes altres dades es compararien amb els d'ara."""
    X, y = _table()
    _studied(tmp_path, monkeypatch, X, y)
    with pytest.raises(Refusal, match="amb un altre dataset"):
        _continued(_searcher({"value": (3, 1, 2)}, []), X.columns, "un altre")


def test_a_search_made_on_another_space_or_fixed_numbers_is_not_continued(
    tmp_path, monkeypatch
):
    """Uns trials d'un altre espai o d'altres números no es comparen amb els d'ara."""
    X, y = _table()
    _studied(tmp_path, monkeypatch, X, y)
    searcher = vars(_searcher({"value": (3, 1, 2)}, []))
    for change, why in (
        ({"SPACE": {"value": (3, 1, 2, 4)}}, "sobre un altre espai"),
        ({"FIXED": {"patience": 2}}, "amb uns altres números fixos"),
    ):
        with pytest.raises(Refusal, match=why):
            _continued(types.SimpleNamespace(**searcher | change), X.columns)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_a_trial_without_a_finite_score_diverges_and_keeps_its_dataset(tmp_path, score):
    """Optuna acceptava un infinit, i el classificava; i un trial que divergeix compta.

    Comptat, un trial fet sobre unes altres dades no pot continuar el study.
    """
    X, y = _table()
    searcher = _searcher({"value": (3, 1, 2)}, [])
    store = tmp_path / "seed_0.db"
    study = opened(store, "fake", searcher, 0, X.columns, FINGERPRINTS["dataset"])
    optuna_trial, drawn = asked(study, searcher, "fake")
    assert told(study, optuna_trial, {"value": drawn}, score, FINGERPRINTS)
    assert [one.state.name for one in study.trials] == ["FAIL"]
    with pytest.raises(Refusal, match="amb un altre dataset"):
        opened(store, "fake", searcher, 0, X.columns, "un altre")


def test_a_seed_fits_the_best_of_its_search_and_the_date_probe_the_best_alone(
    tmp_path,
):
    """La sonda de datació en guarda la millor: amb més, no seria la sonda que diu."""
    X, y = _table()
    study = _study(
        tmp_path / "seed_0.db",
        _searcher({"value": (4, 1, 3, 2)}, []),
        X,
        y,
        split(0, PERIOD, DELETED),
        0,
        4,
    )
    found = record(study, "fake", 0, True)
    best = configurations(found, 2)
    assert [one["rank"] for one in best] == [0, 1]
    assert [one["numbers"]["value"] for one in best] == [1, 2]
    assert all(one["trials"] == 4 for one in best)
    assert [one["numbers"]["value"] for one in configurations(found, 1)] == [1]


def test_a_family_never_searched_fits_once_with_no_numbers():
    assert configurations(None, 0) == [{"rank": 0}]
