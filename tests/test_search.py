"""La cerca: l'espai de les xarxes, les files que veu una prova, una cerca represa, i quin
registre llegeix una passada.

Cap d'aquestes coses no peta quan falla. Una L1 de paràmetre llegida per posició
penalitzaria el paràmetre equivocat; una prova que veiés la prova en triaria els números;
una cerca represa que repetís les primeres proves en faria menys de les que diu; i un
registre llegit per una altra llavor, o fet sobre unes altres files, donaria a una passada
números que no són seus.
"""
import datetime

import numpy as np
import optuna
import pandas as pd
import pytest

from epf import paths, search
from epf.models.ddnn import search as ddnn_search
from epf.models.ddnn.distributions import JOHNSON_SU, POINT
from epf.split import split

# Una xarxa de l'espai, sense cap L1 als pesos ni a les sortides de les capes amagades.
FIXED = {"neurons_1": 32, "neurons_2": 16, "activation_1": "relu", "activation_2": "elu",
         "dropout": False, "regularize_h1_kernel": False, "regularize_h2_kernel": False,
         "regularize_h1_activation": False, "regularize_h2_activation": False,
         "learning_rate": 1e-3}

# Un període curt de setmanes senceres, de dimarts a dilluns, amb una setmana suprimida.
PERIOD = (datetime.date(2019, 1, 1), datetime.date(2019, 7, 29))
DELETED = (datetime.date(2019, 4, 30), datetime.date(2019, 5, 6))


def test_each_parameters_l1_is_the_one_named_after_it():
    # El codi de l'article posa el pes de la cua de Johnson SU abans de l'asimetria: una
    # L1 presa per posició penalitzaria l'un amb la de l'altre.
    rates = {"location": 1e-4, "scale": 1e-3, "skewness": 1e-2, "tail_weight": 1e-1}
    fixed = FIXED | {f"regularize_{name}": True for name in rates}
    fixed |= {f"{name}_l1": rate for name, rate in rates.items()}
    tuned = ddnn_search.space(optuna.trial.FixedTrial(fixed), JOHNSON_SU)
    assert tuned["parameter_l1"] == [rates[name] for name in JOHNSON_SU.names]


def test_the_point_networks_space_asks_for_no_output_penalty():
    # La xarxa de punt de l'article no té cap L1 a la sortida: si se'n demanés una,
    # `FixedTrial` petaria per un paràmetre que no té.
    tuned = ddnn_search.space(optuna.trial.FixedTrial(FIXED), POINT, penalised=False)
    assert tuned["parameter_l1"] == [0.0]


def _table():
    days = pd.date_range(*PERIOD, freq="D")
    return (pd.DataFrame({"x": np.arange(len(days), dtype=float)}, index=days),
            pd.Series(np.zeros(len(days)), index=days))


def _searcher(grid, seen):
    """Un cercador que no ajusta res: desa els dies que veu cada prova, i la puntuació és
    la configuració mateixa."""
    def trial_of(trial, train, validate, seed):
        seen.append((train[0].index, validate[0].index))
        if grid is None:
            value = trial.suggest_float("value", 0.0, 1.0)
        else:
            value = trial.suggest_categorical("value", grid["value"])
        return {"value": value}, float(value)
    return search.Searcher("daily", grid, trial_of)


def test_a_trial_sees_the_training_and_validation_days_and_no_test_day(tmp_path):
    X, y = _table()
    parts = split(0, PERIOD, DELETED)
    seen = []
    search.run(tmp_path / "study.db", "fake", _searcher({"value": tuple(range(8))}, seen),
               X, y, parts, 0, 3)
    assert len(seen) == 3
    for train, validate in seen:
        assert train.equals(parts.training) and validate.equals(parts.validation)
        assert not train.isin(parts.test).any() and not validate.isin(parts.test).any()


@pytest.mark.parametrize("grid", [{"value": tuple(range(8))}, None],
                         ids=["graella", "continu"])
def test_a_resumed_search_does_not_repeat_its_trials(tmp_path, grid):
    # Amb la llavor de sempre a cada procés, el mostrejador d'un espai continu tornaria a
    # proposar les tres primeres.
    X, y = _table()
    parts = split(0, PERIOD, DELETED)
    searcher = _searcher(grid, [])
    search.run(tmp_path / "study.db", "fake", searcher, X, y, parts, 0, 3)
    study = search.run(tmp_path / "study.db", "fake", searcher, X, y, parts, 0, 6)
    values = [one.params["value"] for one in search.completed(study)]
    assert len(values) == 6 and len(set(values)) == 6


def _saved(tmp_path, monkeypatch, X, y):
    """Una cerca de dues proves de la família `fake` a la llavor 0, desada a l'assaig."""
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    where = search.folder("smoke", "fake")
    where.mkdir(parents=True)
    study = search.run(where / "seed_0.db", "fake", _searcher({"value": (3, 1, 2)}, []),
                       X, y, split(0, PERIOD, DELETED), 0, 2)
    found = search.record(study, "fake", 0, "smoke", "commit")
    search.save(found, "smoke")
    return found


def test_a_saved_search_is_read_for_its_own_family_and_seed_alone(tmp_path, monkeypatch):
    X, y = _table()
    found = _saved(tmp_path, monkeypatch, X, y)
    assert search.searched("smoke", "fake", 0, X.columns) == found
    assert [one["score"] for one in found["best"]] == sorted(
        one["score"] for one in found["best"])
    assert search.searched("smoke", "fake", 1, X.columns) is None
    assert search.searched("smoke", "other", 0, X.columns) is None
    assert search.searched("measurement", "fake", 0, X.columns) is None


def test_a_search_made_on_other_columns_is_neither_read_nor_resumed(tmp_path, monkeypatch):
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    other = X.assign(z=0.0)
    with pytest.raises(ValueError):
        search.searched("smoke", "fake", 0, other.columns)
    with pytest.raises(ValueError):
        search.run(search.folder("smoke", "fake") / "seed_0.db", "fake",
                   _searcher({"value": (3, 1, 2)}, []), other, y,
                   split(0, PERIOD, DELETED), 0, 3)
