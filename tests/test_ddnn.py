"""L'atribució de les xarxes distribucionals: cada paràmetre la seva, i que sumi."""
import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.models.ddnn.ddnn import attribute_rows, fit_rows, parameters_rows
from epf.models.ddnn.distributions import JOHNSON_SU, NORMAL

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
