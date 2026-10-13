"""La cerca de la xarxa: el que s'hi pot trencar en silenci.

Una prova que s'ajustés amb algun dia de prova puntuaria massa bé, la cerca triaria els
números que més hi filtren, i el model que en sortís seria pitjor que el que diu el seu
MAE. Una cerca represa que repetís les seves primeres proves gastaria ajustos sense
provar res de nou. I una cerca desada que el camí no llegís deixaria la xarxa sobre els
números copiats. Cap de les tres no dóna cap error.
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


def test_a_saved_search_replaces_the_copied_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SEARCHES", tmp_path)
    copied = dnn_hyperparameters.searched()
    assert copied["unsearched"] is True
    assert copied["seed"] == COPIED["seed"]
    chosen = COPIED | {"seed": 7, "activation": "tanh"}
    dnn_hyperparameters.save(INPUTS[0], {"trials": 3, "hyperparameters": chosen})
    found = dnn_hyperparameters.searched()
    assert "unsearched" not in found
    assert (found["seed"], found["activation"]) == (7, "tanh")
    assert found["search"]["trials"] == 3


def test_a_resumed_search_does_not_repeat_its_first_trials(tmp_path):
    # Mesurat el 2026-10-13 en una cerca de prova sobre la matriu real: amb la llavor
    # fixa, la primera prova de la represa era la mateixa que la primera de totes.
    store = tmp_path / "dnn.db"

    def objective(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    study(store).optimize(objective, n_trials=2)
    resumed = study(store)
    resumed.optimize(objective, n_trials=1)
    assert resumed.trials[2].params != resumed.trials[0].params
