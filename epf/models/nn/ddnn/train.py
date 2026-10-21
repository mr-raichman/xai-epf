"""La passada de les xarxes distribucionals de Marcjasz et al. (2023).

Una xarxa per distribució i seed, on l'article en mitjana quatre.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import PRICE_UNITS, TOLERANCE, UNITLESS, group
from epf.contract.metrics import scored_distribution
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.nn import network
from epf.models.nn.ddnn import attribution
from epf.models.nn.ddnn.distributions import DISTRIBUTIONS
from epf.models.nn.hyperparameters import FIXED
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# Cada família, com la planifica una passada: tots els números fixos que llegeixen els
# seus trials i els seus ajustos, i la millor configuració de cada seed de la seva
# pròpia cerca.
FAMILIES = {
    name: {"fixed": FIXED, "kept": 1, "searched_as": name, "residuals_checked": True}
    for name in DISTRIBUTIONS
}


def _fitted(distribution, X, y, parts, explained, seed, numbers):
    """Ajusta la xarxa d'una configuració i n'explica les files explicades.

    Explica cada paràmetre contra els dies d'entrenament, i puntua la distribució de les
    setmanes de prova.
    """
    training, validation, test, shown = (
        X.index.isin(days) for days in (*parts, explained)
    )
    fitted = network.fit_rows(
        distribution,
        (X[training], y[training]),
        (X[validation], y[validation]),
        numbers,
        seed,
    )
    explanation = attribution.attribute_rows(
        distribution, fitted, X[shown], background=X[training]
    )
    # La base més les contribucions torna cada paràmetre, i la base és la mitjana de
    # cada paràmetre sobre tots els dies d'entrenament. La mitjana, en `float64`: en
    # `float32`, la dels paràmetres d'un miler de dies se'n va més que el marge.
    gap = max(
        np.abs(
            explanation.base
            + explanation.values.sum(axis=1)
            - network.parameters_rows(distribution, fitted, X[shown])
        ).max(),
        np.abs(
            explanation.base
            - network.parameters_rows(distribution, fitted, X[training]).mean(
                axis=0, dtype=np.float64
            )
        ).max(),
    )
    found = network.parameters_rows(distribution, fitted, X[test])
    scores = scored_distribution(
        y[test], lambda probabilities: distribution.quantiles(found, probabilities)
    )
    return Fitted(
        distribution.point(found).ravel(),
        y[test].to_numpy().ravel(),
        {
            "names": np.array(distribution.names),
            "columns": np.array(explanation.columns),
            "groups": np.array([group(column) for column in explanation.columns]),
            "inputs": explanation.inputs,
            # La localització i l'escala en EUR/MWh, la resta sense unitats.
            "units": np.array(
                [PRICE_UNITS, PRICE_UNITS, UNITLESS, UNITLESS][
                    : len(distribution.names)
                ]
            ),
            "values": explanation.values,
            "base": explanation.base,
            "parameters": found,
        },
        gap,
        TOLERANCE,
        scores,
        stored=attribution.sums,
    )


def train(family, folder, seed, configuration, explained_days):
    """Ajusta, comprova, puntua i desa una configuració del pla d'una seed.

    `explained_days` són els dies de prova que s'expliquen, tots amb `None`.
    """
    X, y = matrix.matrices(matrix.series()).daily
    parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
    explained = parts.test if explained_days is None else parts.test[:explained_days]
    rank = configuration["rank"]
    fitted = _fitted(
        DISTRIBUTIONS[family], X, y, parts, explained, seed, configuration["numbers"]
    )
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, explained, fitted, rank)
