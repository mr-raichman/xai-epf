"""L'atribució de la xarxa elàstica: el valor de SHAP exacte d'un model lineal, a
l'espai transformat de Lago et al. (2021), i les sumes que se'n tornen a fer sobre el
fitxer desat.
"""

import collections

import numpy as np
import pandas as pd

from epf.contract.explanation import summed
from epf.contract.matrix import HOUR
from epf.models.elastic_net.regression import by_hour, design

# L'atribució de cada fila, a l'espai transformat: les contribucions per columna, la
# base de l'hora de la fila i la predicció de la fila al mateix espai, que és la que
# base més contribucions ha de tornar.
Explanation = collections.namedtuple("Explanation", "values base transformed")


def attribute_rows(model, X):
    """Les contribucions de cada fila, el valor de SHAP exacte d'un model lineal.

    La d'una columna és el seu coeficient per la distància a la mitjana de les files
    d'ajust de la seva hora. Són a l'espai transformat i no en EUR/MWh: `sinh` no és
    lineal i no en surt cap versió exacta. Una columna que el model d'una hora no porta
    hi té zero.
    """
    columns = [column for column in X.columns if column != HOUR]
    values = np.zeros((len(X), len(columns)))
    base, transformed = np.empty(len(X)), np.empty(len(X))
    for fitted, rows in zip(model, by_hour(X), strict=True):
        if not len(rows):
            continue
        matrix = design(X.iloc[rows], fitted)
        coefficients = fitted.model.coef_
        positions = [columns.index(column) for column in fitted.columns]
        values[np.ix_(rows, positions)] = coefficients * (matrix - fitted.background)
        base[rows] = fitted.model.intercept_ + coefficients @ fitted.background
        transformed[rows] = fitted.model.predict(matrix)
    return Explanation(
        pd.DataFrame(values, index=X.index, columns=columns),
        pd.Series(base, index=X.index),
        pd.Series(transformed, index=X.index),
    )


def sums(found):
    """La suma de la xarxa elàstica sobre el seu fitxer, a l'espai transformat."""
    return [(*summed(found["base"], found["values"], 1), found["transformed"])]
