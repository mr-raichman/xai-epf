"""La passada de l'elastic net d'Uniejewski et al. (2016)."""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import (
    TRANSFORMED_TOLERANCE,
    TRANSFORMED_UNITS,
    group,
)
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.elastic_net import attribution, regression
from epf.models.elastic_net.hyperparameters import MAX_ITER, NAME
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La família, com la planifica una passada: el sostre d'iteracions, que no es cerca, i
# les `BEST` millors de cada seed de la seva pròpia cerca.
FAMILIES = {
    NAME: {
        "fixed": {"max_iter": MAX_ITER},
        "kept": "BEST",
        "searched_as": NAME,
    }
}


def _fitted(X, y, parts, explained, numbers):
    """Ajusta els 24 models d'una configuració i n'explica les files explicades.

    El càstig i la barreja són els de `numbers`.
    """
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


def train(family, folder, seed, configuration, explained_days):
    """Ajusta, comprova, puntua i desa una configuració del pla d'una seed.

    `explained_days` són els dies de prova que s'expliquen, tots amb `None`.
    """
    X, y = matrix.matrices(matrix.series()).hourly
    parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
    explained = parts.test if explained_days is None else parts.test[:explained_days]
    rank = configuration["rank"]
    fitted = _fitted(X, y, parts, explained, configuration["numbers"])
    path = folder / paths.SEED.format(seed=seed, rank=rank)
    save_model(path, parts, explained, fitted, rank)
