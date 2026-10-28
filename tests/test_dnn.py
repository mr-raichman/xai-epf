import keras
import numpy as np
import pandas as pd
import pytest

from epf.contract.matrix import DAY_OF_WEEK_COLUMNS
from epf.contract.split import split
from epf.models.dnn import network, search
from epf.models.dnn import train as dnn
from epf.models.dnn.attribution import attribute_rows
from epf.parameters import DEVELOPMENT, HOUR_COLUMNS

# Una xarxa petita i un entrenament curt: el que es comprova és la suma, que no depèn de
# com d'ajustada estigui.
TUNED = {
    "hidden": [8, 8],
    "activation": ["relu", "softplus"],
    "learning_rate": 0.01,
    "batch": 16,
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


def test_the_base_and_contributions_add_up_to_the_prediction():
    # La base i les contribucions de DeepExplainer contra la predicció de la xarxa, a
    # través de la normalització per lots de l'entrada.
    X, y = rows(150)
    fitted, _ = network.fit_rows(
        (X[:120], y[:120]), (X[120:], y[120:]), TUNED, 0, epochs=5, patience=2
    )
    explained = attribute_rows(fitted, X[:5], background=X[:120])
    assert explained.values.shape == (5, 7, 24)
    rebuilt = explained.base + explained.values.sum(axis=1)
    assert np.allclose(rebuilt, network.predict_rows(fitted, X[:5]), atol=1e-3)


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
    parts = split(3, DEVELOPMENT, {}, 1)
    for fit in (
        lambda: search.trial(dnn.NAME, TUNED, (X, y), (X, y), 3),
        lambda: dnn._fitted(X, y, parts, parts.test, 3, TUNED, None),
    ):
        with pytest.raises(Stopped):
            fit()
    assert seen == [3, 3]


def test_a_fit_waits_as_many_weight_updates_at_any_batch(monkeypatch):
    """A batch 256, 50 èpoques aturaven una xarxa encara al seu altiplà inicial."""
    assert [network.patience_at(batch) for batch in (32, 64, 128, 256)] == [
        50,
        100,
        200,
        400,
    ]
    seen = []

    class Recorded(keras.callbacks.EarlyStopping):
        def __init__(self, **keywords):
            seen.append(keywords["patience"])
            super().__init__(**keywords)

    monkeypatch.setattr(keras.callbacks, "EarlyStopping", Recorded)
    X, y = rows(40)
    network.fit_rows(
        (X[:30], y[:30]), (X[30:], y[30:]), TUNED | {"batch": 64}, 0, epochs=1
    )
    assert seen == [100]


@pytest.mark.parametrize(
    "changed, why", [("EPOCHS", "ceiling"), ("PATIENCE", "patience")]
)
def test_a_fits_stop_follows_from_the_epoch_it_keeps(monkeypatch, changed, why):
    """Es desa la millor època: les que es fan, i per què, se'n dedueixen."""
    # Un sostre de 3 èpoques atura abans que la paciència a batch 32; una paciència de
    # 2, abans que el sostre.
    monkeypatch.setattr(network, changed, {"EPOCHS": 3, "PATIENCE": 2}[changed])
    X, y = rows(150)
    tuned = TUNED | {"batch": 32}
    fitted, epochs = network.fit_rows(
        (X[:120], y[:120]), (X[120:], y[120:]), tuned, 0, epochs=network.EPOCHS
    )
    found, ran = network.stopped(epochs, tuned["batch"])
    assert found == why
    assert ran == len(fitted.history.history["val_loss"])
