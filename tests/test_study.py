"""Les cerques i el que en llegeix una passada: les files que veu una prova, una cerca
represa, quin registre es llegeix, i quines configuracions s'ajusten a cada llavor.

Res d'això no peta quan falla. Una prova que veiés la prova en triaria els números; una
cerca represa que repetís les primeres proves en faria menys de les que diu; un registre
llegit per una altra llavor, o fet sobre unes altres files, donaria a una passada números
que no són seus; una cerca a mitges barrejaria llavors cercades i sense cercar en una sola
dispersió; i la prova de datació amb totes les configuracions en lloc de la millor no
seria la prova que diu.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from epf import describe, paths, search
from epf.contract.split import split
from epf.contract.study import completed, configurations, folder, record, run, save, searched

# Un període curt de setmanes senceres, de dimarts a dilluns, amb una setmana suprimida.
PERIOD = (datetime.date(2019, 1, 1), datetime.date(2019, 7, 29))
DELETED = (datetime.date(2019, 4, 30), datetime.date(2019, 5, 6))


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
    run(tmp_path / "study.db", "fake", _searcher({"value": tuple(range(8))}, seen),
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
    run(tmp_path / "study.db", "fake", searcher, X, y, parts, 0, 3)
    study = run(tmp_path / "study.db", "fake", searcher, X, y, parts, 0, 6)
    values = [one.params["value"] for one in completed(study)]
    assert len(values) == 6 and len(set(values)) == 6


def _saved(tmp_path, monkeypatch, X, y):
    """Una cerca de dues proves de la família `fake` a la llavor 0, desada a l'assaig."""
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    where = folder("smoke", "fake")
    where.mkdir(parents=True)
    study = run(where / "seed_0.db", "fake", _searcher({"value": (3, 1, 2)}, []),
                       X, y, split(0, PERIOD, DELETED), 0, 2)
    found = record(study, "fake", 0, "smoke", "commit")
    save(found, "smoke")
    return found


def test_a_saved_search_is_read_for_its_own_family_and_seed_alone(tmp_path, monkeypatch):
    X, y = _table()
    found = _saved(tmp_path, monkeypatch, X, y)
    assert searched("smoke", "fake", 0, X.columns) == found
    assert [one["score"] for one in found["best"]] == sorted(
        one["score"] for one in found["best"])
    assert searched("smoke", "fake", 1, X.columns) is None
    assert searched("smoke", "other", 0, X.columns) is None
    assert searched("measurement", "fake", 0, X.columns) is None


def test_a_search_made_on_other_columns_is_neither_read_nor_resumed(tmp_path, monkeypatch):
    X, y = _table()
    _saved(tmp_path, monkeypatch, X, y)
    other = X.assign(z=0.0)
    with pytest.raises(ValueError):
        searched("smoke", "fake", 0, other.columns)
    with pytest.raises(ValueError):
        run(folder("smoke", "fake") / "seed_0.db", "fake",
                   _searcher({"value": (3, 1, 2)}, []), other, y,
                   split(0, PERIOD, DELETED), 0, 3)


# Una família que es cerca i se n'ajusten les tres millors configuracions de cada llavor.
FAKE = describe.Describer("daily", (), None, {}, {"value": 0}, 3)


def _searched(tmp_path, monkeypatch, seeds):
    """Cerques de quatre proves de la família `fake` a l'assaig, per a `seeds`, i les
    columnes sobre les quals es van fer."""
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    X, y = _table()
    where = folder("smoke", "fake")
    where.mkdir(parents=True)
    for seed in seeds:
        study = run(where / f"seed_{seed}.db", "fake",
                           _searcher({"value": (4, 1, 3, 2)}, []), X, y,
                           split(seed, PERIOD, DELETED), seed, 4)
        save(record(study, "fake", seed, "smoke", "commit"), "smoke")
    return X.columns


def test_with_no_seed_searched_each_seed_fits_the_placeholders_marked(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, [])
    found = configurations("smoke", "fake", FAKE, columns, (0, 1), False)
    assert found == {seed: [{"rank": 0, "numbers": {"value": 0}, "unsearched": True}]
                     for seed in (0, 1)}


def test_with_every_seed_searched_each_fits_its_best_and_the_date_probe_the_first(
        tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0, 1))
    found = configurations("smoke", "fake", FAKE, columns, (0, 1), False)
    for seed in (0, 1):
        assert [one["rank"] for one in found[seed]] == [0, 1, 2]
        assert [one["numbers"]["value"] for one in found[seed]] == [1, 2, 3]
        assert all(one["trials"] == 4 for one in found[seed])
    alone = configurations("smoke", "fake", FAKE, columns, (0, 1), True)
    assert {seed: [one["numbers"]["value"] for one in ones]
            for seed, ones in alone.items()} == {0: [1], 1: [1]}


def test_a_search_of_some_seeds_only_is_refused(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    with pytest.raises(ValueError):
        configurations("smoke", "fake", FAKE, columns, (0, 1), False)


def test_a_family_never_searched_fits_once_with_no_numbers(tmp_path, monkeypatch):
    columns = _searched(tmp_path, monkeypatch, (0,))
    never = FAKE._replace(placeholder=None, kept=0)
    assert configurations("smoke", "fake", never, columns, (0, 1), False) == {
        0: [{"rank": 0}], 1: [{"rank": 0}]}
