"""La passada de les xarxes distribucionals de Marcjasz et al. (2023), una distribució
cada vegada, de la matriu diària als seus fitxers.

Una família per distribució, que és el que fa l'article. **La predicció puntual és la
mediana de la distribució**, la que l'article fa servir per al MAE. Els paràmetres es
desen a part.

Per a cada llavor: la xarxa de la configuració ajustada amb les files d'entrenament i
aturada sobre tota la validació, els paràmetres de les setmanes de prova, la puntuació
del punt i de la distribució, l'atribució de cada paràmetre de les files que s'expliquen
contra els dies d'entrenament, la comprovació que suma, i el fitxer.

Una xarxa per distribució i llavor, on Marcjasz et al. (2023) en mitjanen quatre.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import PRICE_UNITS, TOLERANCE, UNITLESS, check, group
from epf.contract.metrics import scored, scored_distribution
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.nn import network
from epf.models.nn.ddnn import attribution
from epf.models.nn.ddnn.distributions import DISTRIBUTIONS
from epf.models.nn.hyperparameters import FIXED, copied
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# Cada família, com la planifica una passada: el lot, les èpoques i la paciència, que no
# es cerquen, els números copiats de l'article mentre no hi ha cerca, i la millor
# configuració de cada llavor de la seva pròpia cerca.
FAMILIES = {
    name: {"fixed": FIXED, "placeholder": copied(name), "kept": 1, "searched_as": name}
    for name in DISTRIBUTIONS
}


def _fitted(distribution, X, y, parts, explained, numbers):
    """La xarxa d'una distribució, l'atribució de cada paràmetre de les files
    explicades, i la distribució de les setmanes de prova puntuada."""
    training, validation, test, shown = (
        X.index.isin(days) for days in (*parts, explained)
    )
    fitted = network.fit_rows(
        distribution,
        (X[training], y[training]),
        (X[validation], y[validation]),
        numbers,
    )
    explanation = attribution.attribute_rows(
        distribution, fitted, X[shown], background=X[training]
    )
    # Dues sumes: la base més les contribucions torna cada paràmetre, i la base és la
    # mitjana de cada paràmetre sobre tots els dies d'entrenament.
    gap = max(
        np.abs(
            explanation.base
            + explanation.values.sum(axis=1)
            - network.parameters_rows(distribution, fitted, X[shown])
        ).max(),
        np.abs(
            explanation.base
            - network.parameters_rows(distribution, fitted, X[training]).mean(axis=0)
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


def train(family, folder, configurations, explained_days):
    """La passada d'una distribució: cada llavor ajustada, comprovada, puntuada i
    desada. Torna les parelles `[llavor, configuració]` que ha desat ara."""
    X, y = matrix.matrices(matrix.series()).daily
    distribution = DISTRIBUTIONS[family]
    done = []
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        explained = (
            parts.test if explained_days is None else parts.test[:explained_days]
        )
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            fitted = _fitted(
                distribution, X, y, parts, explained, configuration["numbers"]
            )
            check(fitted, rank)
            scores = scored(fitted.predicted, fitted.actual) | fitted.scores
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
    return done
