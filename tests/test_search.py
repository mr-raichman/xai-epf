"""La cerca de la xarxa: el que s'hi pot trencar en silenci.

Una prova que s'ajustés amb algun dia de prova puntuaria massa bé, la cerca triaria els
números que més hi filtren, i el model que en sortís seria pitjor que el que diu el seu
MAE. Una cerca represa que repetís les seves primeres proves gastaria ajustos sense
provar res de nou, i quatre membres que cerquessin amb la mateixa llavor serien la
mateixa xarxa quatre vegades. I una cerca desada que el camí no llegís, o llegís per a
un altre membre, deixaria la xarxa sobre uns números que no són els seus. Cap d'aquestes
coses no dóna cap error.
"""
import datetime

import pandas as pd

from epf import paths
from epf.models.dnn import dnn_hyperparameters
from epf.models.dnn.dnn_hyperparameters import COPIED, WINDOW
from epf.models.dnn.search import FIRST_TEST_DAY, study, window
from epf.parameters import INPUTS


def test_a_trial_fits_the_four_years_before_the_first_test_day_and_none_of_it():
    index = pd.date_range(FIRST_TEST_DAY - datetime.timedelta(days=WINDOW + 30),
                          periods=WINDOW + 100, freq="D")
    frame = pd.DataFrame(1.0, index=index, columns=["a"])
    features, _ = window(frame, frame["a"])
    assert len(features) == WINDOW
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
