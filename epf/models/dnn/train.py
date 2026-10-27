"""La passada de la xarxa neuronal."""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import PRICE_UNITS, TOLERANCE, group, summed
from epf.contract.record import Fitted, priced, save_model
from epf.models.dnn import attribution, network
from epf.models.dnn.hyperparameters import FIXED

# Nom del model i de la seva família.
NAME = "dnn"

# La família, com la planifica una passada: tots els números fixos que llegeixen els
# seus trials i els seus ajustos, i les `BEST` millors de cada seed de la seva pròpia
# cerca.
FAMILIES = {
    NAME: {
        "fixed": FIXED,
        "kept": "BEST",
        "searched_as": NAME,
        "ljung_box_tested": True,
    }
}


def _fitted(X, y, parts, explained, seed, numbers, information_set):
    """Ajusta la xarxa d'una configuració i n'explica les files explicades.

    Explica el preu de cada hora contra els dies d'entrenament, cada columna en el seu
    grup de l'information set resolt `information_set`.
    """
    training, validation, test, shown = (
        X.index.isin(days) for days in (*parts, explained)
    )
    fitted = network.fit_rows(
        (X[training], y[training]),
        (X[validation], y[validation]),
        numbers,
        seed,
    )
    explanation = attribution.attribute_rows(fitted, X[shown], background=X[training])
    # La base més les contribucions torna la predicció de cada fila, i la base és la
    # mitjana de la predicció sobre tots els dies d'entrenament. Tot en `float64`: en
    # `float32`, la mitjana d'un miler de dies se'n va més que el marge.
    total, _ = summed(explanation.base, explanation.values, 1)
    gap = max(
        np.abs(total - network.predict_rows(fitted, X[shown])).max(),
        np.abs(
            explanation.base
            - network.predict_rows(fitted, X[training]).mean(axis=0, dtype=np.float64)
        ).max(),
    )
    return Fitted(
        network.predict_rows(fitted, X[test]).ravel(),
        y[test].to_numpy().ravel(),
        {
            "columns": np.array(explanation.columns),
            "groups": np.array(
                [group(column, information_set) for column in explanation.columns]
            ),
            "inputs": explanation.inputs,
            "unit": PRICE_UNITS,
            "values": explanation.values,
            "base": explanation.base,
        },
        gap,
        TOLERANCE,
        stored=attribution.sums,
    )


def train(family, folder, seed, configuration, explained_days, built):
    """Ajusta, comprova, puntua i desa una configuració del pla d'una seed.

    `explained_days` són els dies de prova que s'expliquen, tots amb `None`, i `built`
    les matrius de la passada.
    """
    X, y = built.daily
    parts = matrix.partition(built, seed)
    explained = parts.test if explained_days is None else parts.test[:explained_days]
    rank = configuration["rank"]
    fitted = priced(
        _fitted(
            X, y, parts, explained, seed, configuration["numbers"], built.information
        ),
        built,
        parts.test,
    )
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, explained, fitted, rank)
