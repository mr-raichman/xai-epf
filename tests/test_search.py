"""La cerca de la xarxa: el que s'hi pot trencar en silenci.

Una prova que s'ajustés amb algun dia de prova puntuaria massa bé, la cerca triaria els
números que més hi filtren, i el model que en sortís seria pitjor que el que diu el seu
MAE, i un dia amb unes hores a cada banda faria el mateix a l'arbre. Una cerca represa
que repetís les seves primeres proves gastaria ajustos sense provar res de nou, quatre
membres que cerquessin amb la mateixa llavor serien la mateixa xarxa quatre vegades, i
una graella que es saltés punts o en repetís deixaria l'arbre sense haver provat el que
diu. I una cerca desada que el camí no llegís, o llegís per a un altre membre, deixaria
un model sobre uns números que no són els seus. Cap d'aquestes coses no dóna cap
error.
"""
import datetime

import pandas as pd

from epf import paths
from epf.models.boosting import boosting_hyperparameters
from epf.models.boosting import search as tree_search
from epf.models.dnn import dnn_hyperparameters
from epf.models.dnn.dnn_hyperparameters import COPIED
from epf.models.dnn.search import study
from epf.models.linear import linear_hyperparameters
from epf.models.linear import search as net_search
from epf.models.search import FIRST_TEST_DAY, parts, window
from epf.parameters import FOUR_YEARS, INPUTS


def test_a_trial_fits_the_four_years_before_the_first_test_day_and_none_of_it():
    index = pd.date_range(FIRST_TEST_DAY - datetime.timedelta(days=FOUR_YEARS + 30),
                          periods=FOUR_YEARS + 100, freq="D")
    frame = pd.DataFrame(1.0, index=index, columns=["a"])
    features, _ = window(frame, frame["a"])
    assert len(features) == FOUR_YEARS
    assert features.index[-1] == FIRST_TEST_DAY - datetime.timedelta(days=1)


def test_a_saved_search_replaces_the_copied_numbers_of_its_own_member(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    copied = dnn_hyperparameters.searched()
    assert copied["unsearched"] is True
    assert [member["seed"] for member in copied["members"]] == [
        one["seed"] for one in COPIED]
    chosen = COPIED[2] | {"seed": 7, "activation": "tanh"}
    dnn_hyperparameters.save(INPUTS[0], 2, {"trials": 3, "hyperparameters": chosen})
    found = dnn_hyperparameters.searched()["members"]
    assert "unsearched" not in found[2]
    assert (found[2]["seed"], found[2]["activation"]) == (7, "tanh")
    assert found[2]["search"]["trials"] == 3
    assert [member["seed"] for member in found[:2] + found[3:]] == [
        one["seed"] for one in COPIED[:2] + COPIED[3:]]
    assert all(member["unsearched"] for member in found[:2] + found[3:])


def test_a_resumed_search_does_not_repeat_its_first_trials(tmp_path):
    # Mesurat el 2026-10-13 en una cerca de prova sobre la matriu real: amb la llavor
    # fixa, la primera prova de la represa era la mateixa que la primera de totes.
    store = tmp_path / "dnn.db"

    def objective(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    study(store, 0).optimize(objective, n_trials=2)
    resumed = study(store, 0)
    resumed.optimize(objective, n_trials=1)
    assert resumed.trials[2].params != resumed.trials[0].params


def test_each_member_searches_from_its_own_seed(tmp_path):
    # L'article obté les seves quatre xarxes repetint la cerca amb una llavor inicial
    # diferent; amb la mateixa, les quatre proposarien les mateixes proves.
    def objective(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    first = []
    for member in (0, 1):
        searching = study(tmp_path / f"dnn_{member}.db", member)
        searching.optimize(objective, n_trials=1)
        first.append(searching.trials[0].params)
    assert first[0] != first[1]


def test_a_per_hour_trials_validation_takes_whole_days():
    # La taula per hora porta 24 files per dia; partir-les per files posaria unes hores
    # d'un dia a cada banda, i l'aturada miraria hores veïnes de les que ha après.
    # 41 dies: la quarta part de les files no cau en una frontera de dia, de manera
    # que una partició per files en partiria un.
    days = pd.date_range("2024-01-01", periods=41, freq="D")
    X = pd.DataFrame({"hour": list(range(24)) * len(days)}, index=days.repeat(24))
    train, validate = parts(X)
    assert set(X.index[train]).isdisjoint(X.index[validate])
    assert list(X.index[validate].unique()) == list(days[-10:])
    assert validate.sum() == 10 * 24


def test_the_grid_is_tried_once_each(tmp_path):
    # La cerca en graella del codi d'on surt prova cada punt una vegada. Amb un
    # mostrejador que repetís o se'n saltés, el millor punt no seria el de la graella.
    searching = tree_search.study(tmp_path / "lightgbm.db")
    searching.optimize(lambda trial: sum(tree_search.space(trial).values()),
                       n_trials=tree_search.POINTS)
    seen = [tuple(sorted(trial.params.items())) for trial in searching.trials]
    assert len(seen) == len(set(seen)) == tree_search.POINTS == 288


def test_a_saved_tree_search_replaces_the_unsearched_point(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    assert boosting_hyperparameters.searched()["unsearched"] is True
    chosen = {"num_leaves": 200, "max_depth": 5, "subsample": 0.9,
              "learning_rate": 0.01, "min_child_samples": 100, "trees": 437}
    boosting_hyperparameters.save(INPUTS[0], {"trials": 288, "hyperparameters": chosen})
    found = boosting_hyperparameters.searched()
    assert "unsearched" not in found
    assert {name: found[name] for name in chosen} == chosen
    assert found["search"]["trials"] == 288


def test_the_nets_grid_is_tried_once_each(tmp_path):
    searching = net_search.study(tmp_path / "elastic_net.db")
    searching.optimize(lambda trial: sum(net_search.space(trial)),
                       n_trials=net_search.POINTS)
    seen = [tuple(sorted(trial.params.items())) for trial in searching.trials]
    assert len(seen) == len(set(seen)) == net_search.POINTS == 102


def test_a_saved_net_search_replaces_the_unsearched_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    assert linear_hyperparameters.searched()["unsearched"] is True
    linear_hyperparameters.save(INPUTS[0], {"trials": 102, "hyperparameters":
                                            {"penalty": 0.02, "mixing": 0.25}})
    found = linear_hyperparameters.searched()
    assert "unsearched" not in found
    assert (found["penalty"], found["mixing"]) == (0.02, 0.25)
