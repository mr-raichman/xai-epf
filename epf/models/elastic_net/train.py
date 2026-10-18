"""La passada de la xarxa elàstica d'Uniejewski et al. (2016), de la matriu per hora als
seus fitxers.

Per a cada llavor i configuració: els 24 models ajustats amb les files d'entrenament, el
preu de les setmanes de prova, la seva puntuació, l'atribució de les files que
s'expliquen, la comprovació que suma la predicció que explica, i el fitxer.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import (
    TRANSFORMED_TOLERANCE,
    TRANSFORMED_UNITS,
    check,
    group,
)
from epf.contract.metrics import scored
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.elastic_net import attribution, regression
from epf.models.elastic_net.hyperparameters import MAX_ITER, MIXING, NAME, PENALTY
from epf.parameters import BEST, BLACKOUT_WEEK, DEVELOPMENT

# La família, com la planifica una passada: el sostre d'iteracions, que no es cerca, el
# càstig i la barreja de lloc mentre no hi ha cerca, i les deu millors de cada llavor de
# la seva pròpia cerca.
FAMILIES = {
    NAME: {
        "fixed": {"max_iter": MAX_ITER},
        "placeholder": {"penalty": PENALTY, "mixing": MIXING},
        "kept": BEST,
        "searched_as": NAME,
    }
}


def _fitted(X, y, parts, explained, numbers):
    """Els 24 models d'una configuració, amb el càstig i la barreja de `numbers`, i
    l'atribució de les files explicades."""
    training, test, shown = (
        X.index.isin(days) for days in (parts.training, parts.test, explained)
    )
    model = regression.fit_rows(
        X[training], y[training], numbers["penalty"], numbers["mixing"]
    )
    explanation = attribution.attribute_rows(model, X[shown])
    gap = np.abs(
        explanation.base + explanation.values.sum(axis=1) - explanation.transformed
    ).max()
    return Fitted(
        regression.predict_rows(model, X[test]).to_numpy(),
        y[test].to_numpy(),
        {
            "columns": np.array(explanation.values.columns, dtype=str),
            "groups": np.array(
                [group(column) for column in explanation.values.columns]
            ),
            "inputs": X.loc[shown, explanation.values.columns].to_numpy(),
            "units": np.array([TRANSFORMED_UNITS]),
            "values": explanation.values.to_numpy(),
            "base": explanation.base.to_numpy(),
            "transformed": explanation.transformed.to_numpy(),
        },
        gap,
        TRANSFORMED_TOLERANCE,
        stored=attribution.sums,
    )


def train(family, folder, configurations, explained_days):
    """La passada de la família: cada configuració de cada llavor ajustada, comprovada,
    puntuada i desada. Torna les parelles `[llavor, configuració]` que ha desat ara."""
    X, y = matrix.matrices(matrix.series()).hourly
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
            fitted = _fitted(X, y, parts, explained, configuration["numbers"])
            check(fitted, rank)
            scores = scored(fitted.predicted, fitted.actual)
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
    return done
