"""Les puntuacions de tots els models."""

import numpy as np

from epf import Refusal


def _fail_on_absent(actual):
    """Falla si el preu real té cap valor absent, i diu quants.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés.
    """
    absent = int(np.isnan(np.asarray(actual)).sum())
    # Només un error del codi hi arriba, i n'hi ha prou amb el recompte: el dataset no
    # té cap preu absent dins del seu tram, i `matrix.series` retalla cada sèrie al seu.
    if absent:
        raise ValueError(
            "el preu real té 1 valor absent"
            if absent == 1
            else f"el preu real té {absent} valors absents"
        )


def mae(predicted, actual):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def scored(predicted, actual):
    """Torna l'R² i el MAE de totes les files juntes.

    Falla amb un preu real absent, i refusa una predicció que no és un número finit: un
    sol NaN tornaria NaN totes dues puntuacions sense dir-ho.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    _fail_on_absent(actual)
    wrong = int((~np.isfinite(predicted)).sum())
    if wrong:
        raise Refusal(
            "la predicció té 1 valor no finit"
            if wrong == 1
            else f"la predicció té {wrong} valors no finits"
        )
    error = predicted - actual
    spread = actual - np.mean(actual)
    return {
        "r2": float(1 - np.sum(error**2) / np.sum(spread**2)),
        "mae": mae(predicted, actual),
    }
