"""Les xarxes distribucionals: la seva partició de validació, i l'atribució de cada
paràmetre, que ha de sumar.

La partició és el que pot trencar-se en silenci. Si un dia caigués a les dues bandes,
l'aturada anticipada miraria dies amb què la xarxa ja s'ha entrenat, l'entrenament
duraria més del que toca i l'error de validació sortiria massa bo, sense que res petés.
"""
import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.models.ddnn.ddnn import attribute_rows, fit_rows, parameters_rows, split
from epf.models.ddnn.distributions import JOHNSON_SU, NORMAL
from epf.parameters import DAYS_PER_WEEK

# Una xarxa petita i un entrenament curt: el que es comprova és la suma, que no depèn
# de com d'ajustada estigui.
TUNED = {"hidden": [8], "activation": ["relu"], "dropout": 0.0,
         "batch_normalisation": False, "l1": [0.0], "initialiser": "glorot_uniform",
         "learning_rate": 0.01}


def rows(count, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(count, 6)), columns=[f"x{n}" for n in range(6)])
    weights = rng.normal(size=(6, 24))
    y = pd.DataFrame(40 + 15 * X.to_numpy() @ weights / 6 + rng.normal(size=(count, 24)),
                     columns=HOUR_COLUMNS)
    return X, y


@pytest.mark.parametrize("distribution", [NORMAL, JOHNSON_SU], ids=["normal", "jsu"])
def test_each_parameters_base_and_contributions_add_up_to_it(distribution):
    # Dos camins independents: la base i les contribucions de DeepExplainer contra la
    # predicció de la xarxa, i el pas a EUR/MWh de l'una contra el de l'altra.
    X, y = rows(150)
    fitted = fit_rows(distribution, (X[:120], y[:120]), (X[120:], y[120:]), TUNED,
                      batch=16, epochs=5, patience=2)
    explained = attribute_rows(distribution, fitted, X[:5], background=X[:120])
    direct = parameters_rows(distribution, fitted, X[:5])
    assert explained.values.shape == (5, 6, 24, len(distribution.names))
    rebuilt = explained.base + explained.values.sum(axis=1)
    assert np.allclose(rebuilt, direct, atol=1e-3)


COUNT = 1453
WEEKS = 42


def parts(count=COUNT, weeks=WEEKS, seed=0):
    return split(count, weeks, np.random.default_rng(seed))


def test_the_two_parts_are_disjoint_and_cover_the_window():
    train, validate = parts()
    assert set(train).isdisjoint(validate)
    assert sorted([*train, *validate]) == list(range(COUNT))


def test_validation_is_exactly_the_weeks_asked_for():
    _, validate = parts()
    assert len(validate) == WEEKS * DAYS_PER_WEEK


def test_the_validation_days_come_from_the_whole_window_and_not_its_tail():
    # Les tria barrejant tota la finestra, de manera que la xarxa s'entreni també amb
    # els dies més recents. Prenent-ne el tram final, el màxim seria l'últim dia i el
    # mínim el primer d'aquell tram.
    _, validate = parts()
    assert min(validate) < COUNT // 4
    assert max(validate) > 3 * COUNT // 4


def test_the_same_seed_gives_the_same_split():
    # Una execució s'ha de poder repetir bit a bit, i la partició hi entra.
    assert np.array_equal(parts()[1], parts()[1])
    assert not np.array_equal(parts(seed=0)[1], parts(seed=1)[1])


def test_a_window_too_short_to_split_fails_loudly():
    with pytest.raises(ValueError):
        parts(count=WEEKS * DAYS_PER_WEEK)
