"""Les xarxes distribucionals com les construeix el codi de Marcjasz et al. (2023): la
seva capa de distribució, i l'atribució de cada paràmetre, que ha de sumar.

Cap d'aquestes coses no peta quan falla. Un pes de la cua que baixés d'1 donaria una
distribució que l'article no permet, i una L1 per paràmetre que no quadrés amb la
distribució faria llegir a la pèrdua menys sortides de les que la xarxa treu.
"""
import numpy as np
import pandas as pd
import pytest

from epf.calendar import HOUR_COLUMNS
from epf.matrix import DAY_OF_WEEK_COLUMNS
from epf.models.ddnn.ddnn import attribute_rows, build, fit_rows, parameters_rows
from epf.models.ddnn.distributions import JOHNSON_SU, NORMAL, parameters

# Una xarxa petita i un entrenament curt: el que es comprova és la suma, que no depèn
# de com d'ajustada estigui.
TUNED = {"hidden": [8, 8], "activation": ["relu", "softplus"], "dropout": 0.1,
         "l1": [0.0, 1e-4], "activity_l1": [1e-4, 0.0], "learning_rate": 0.01}


def tuned(distribution):
    """Els números de prova amb una L1 per paràmetre de la distribució."""
    return TUNED | {"parameter_l1": [1e-4] * len(distribution.names)}


def rows(count, seed=0):
    """Sis columnes i els set indicadors del dia, que la xarxa llegeix en una de sola."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(50, 20, size=(count, 6)),
                     columns=[f"x{n}" for n in range(6)])
    weights = rng.normal(size=(6, 24))
    y = pd.DataFrame(X.to_numpy() @ weights / 6 + rng.normal(size=(count, 24)),
                     columns=HOUR_COLUMNS)
    week = len(DAY_OF_WEEK_COLUMNS)
    days = pd.DataFrame(np.eye(week)[np.arange(count) % week],
                        columns=DAY_OF_WEEK_COLUMNS)
    return pd.concat([X, days], axis="columns"), y


@pytest.mark.parametrize("distribution", [NORMAL, JOHNSON_SU], ids=["normal", "jsu"])
def test_each_parameters_base_and_contributions_add_up_to_it(distribution):
    # La base i les contribucions de DeepExplainer contra la predicció de la xarxa, a
    # través de la normalització per lots de l'entrada i de les L1 de la sortida.
    X, y = rows(150)
    fitted = fit_rows(distribution, (X[:120], y[:120]), (X[120:], y[120:]),
                      tuned(distribution), batch=16, epochs=5, patience=2)
    explained = attribute_rows(distribution, fitted, X[:5], background=X[:120])
    direct = parameters_rows(distribution, fitted, X[:5])
    assert explained.values.shape == (5, 7, 24, len(distribution.names))
    rebuilt = explained.base + explained.values.sum(axis=1)
    assert np.allclose(rebuilt, direct, atol=1e-3)


def test_a_parameter_l1_for_another_distribution_is_refused():
    # Amb quatre caps i una pèrdua de dos paràmetres, la pèrdua en llegiria dos i la
    # xarxa n'aprendria quatre sense que res petés.
    with pytest.raises(ValueError):
        build(NORMAL, 6, tuned(JOHNSON_SU))


def test_the_tail_weight_never_drops_below_one():
    # El pes de la cua de Johnson SU surt de `1 + 3 softplus` al codi de l'article: com
    # més negativa la sortida crua, més a prop d'1, i mai per sota.
    raw = np.full((3, len(JOHNSON_SU.names) * 24), -50.0, dtype="float32")
    tail = parameters(JOHNSON_SU, raw)[..., JOHNSON_SU.names.index("tail_weight")]
    assert (tail >= 1).all()
    assert np.allclose(tail, 1)
