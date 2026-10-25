import numpy as np
import pandas as pd
import pytest

from epf.contract.matrix import DAY_OF_WEEK_COLUMNS
from epf.contract.split import split
from epf.models.dnn import network, search
from epf.models.dnn import train as dnn
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT, HOUR_COLUMNS

# Una xarxa petita i un entrenament curt: el que es comprova és la suma, que no depèn de
# com d'ajustada estigui.
TUNED = {
    "hidden": [8, 8],
    "activation": ["relu", "softplus"],
    "dropout": 0.1,
    "l1": [0.0, 1e-4],
    "activity_l1": [1e-4, 0.0],
    "learning_rate": 0.01,
}


def rows(count, seed=0):
    """Torna `count` files, sis columnes i les dummies del dia, i els seus preus.

    La xarxa llegeix les set dummies en una sola columna.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(50, 20, size=(count, 6)), columns=[f"x{n}" for n in range(6)]
    )
    weights = rng.normal(size=(6, 24))
    y = pd.DataFrame(
        X.to_numpy() @ weights / 6 + rng.normal(size=(count, 24)), columns=HOUR_COLUMNS
    )
    week = len(DAY_OF_WEEK_COLUMNS)
    days = pd.DataFrame(
        np.eye(week)[np.arange(count) % week], columns=DAY_OF_WEEK_COLUMNS
    )
    return pd.concat([X, days], axis="columns"), y


def test_a_trial_and_the_run_hand_the_networks_fit_the_same_seed(monkeypatch):
    """Amb una altra seed, la passada no ajustaria el que la cerca va puntuar."""

    class Stopped(Exception):
        """L'ajust fals s'atura aquí, un cop desada la seed."""

    seen = []

    def recorded(train, validate, tuned, seed, **keywords):
        seen.append(seed)
        raise Stopped

    monkeypatch.setattr(network, "fit_rows", recorded)
    X, y = rows(10)
    parts = split(3, DEVELOPMENT, BLACKOUT_WEEK)
    for fit in (
        lambda: search.trial(dnn.NAME, TUNED, (X, y), (X, y), 3),
        lambda: dnn._fitted(X, y, parts, 3, TUNED),
    ):
        with pytest.raises(Stopped):
            fit()
    assert seen == [3, 3]
