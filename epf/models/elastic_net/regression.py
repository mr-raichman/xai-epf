"""Els models de l'elastic net d'Uniejewski et al. (2016): un model lineal per hora.

Cada hora té el seu model, ajustat només amb les files de la seva hora de la matriu per
hora, que porten les previsions i el programa a aquella hora, com l'article hi porta la
previsió de càrrega. Preus i entrades passen per la transformació de Lago et al. (2021):
la logarítmica de l'article no admet preus a zero o per sota. El càstig és un per a les
24 hores, com a l'article.

Tot és sobre files donades, les d'entrenament d'una seed.
"""

import collections

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from epf.contract.matrix import DAY_OF_WEEK_COLUMNS, HOUR
from epf.models.elastic_net.hyperparameters import MAX_ITER, NAME
from epf.models.elastic_net.scaling import Scaler
from epf.parameters import HOURS_PER_DAY

# El model d'una hora i el que cal per llegir-lo. `columns` són les de la matriu del
# model en el seu ordre, les escalades i després les dummies del dia de la setmana,
# sense cap de les que no es mouen a les files de l'hora. `inputs` escala les entrades,
# `target` l'objectiu i és qui torna a EUR/MWh, i `background` és la mitjana de cada
# columna de la matriu, d'on surten les contribucions.
Hour = collections.namedtuple("Hour", "model columns inputs target background")


def design(rows, fitted):
    """Torna les files com la matriu del model d'una hora, `fitted`.

    Les entrades, transformades, i les dummies del dia de la setmana.
    """
    scaled = len(fitted.inputs.median)
    values = rows[fitted.columns].to_numpy(dtype=float)
    return np.hstack([fitted.inputs.forward(values[:, :scaled]), values[:, scaled:]])


def _prepared(rows, target):
    """Prepara les files d'una hora per al seu model.

    Torna el que cal per llegir el model, encara sense model, la matriu i l'objectiu
    transformat. Una columna que no es mou a les files de l'hora no hi entra, que és la
    previsió solar de nit. La que es mou sense MAD s'escala per la seva desviació
    típica, que és la solar dels vespres d'estiu: la transformació de Lago et al. (2021)
    la refusaria.
    """
    moving = [
        column
        for column in rows.columns
        if column != HOUR and rows[column].nunique() > 1
    ]
    scaled = [column for column in moving if column not in DAY_OF_WEEK_COLUMNS]
    dummies = [column for column in moving if column in DAY_OF_WEEK_COLUMNS]
    inputs = Scaler(rows[scaled].to_numpy(dtype=float), scaled, std_where_flat=True)
    output = Scaler(target.to_numpy(dtype=float)[:, np.newaxis], [target.name])
    fitted = Hour(None, scaled + dummies, inputs, output, None)
    scaled_target = output.forward(target.to_numpy(dtype=float)[:, np.newaxis]).ravel()
    return fitted, design(rows, fitted), scaled_target


def _fit_hour(rows, target, penalty, mixing):
    """Ajusta el model d'una hora amb les seves files i prou."""
    fitted, matrix, scaled_target = _prepared(rows, target)
    model = ElasticNet(alpha=penalty, l1_ratio=mixing, max_iter=MAX_ITER)
    return fitted._replace(
        model=model.fit(matrix, scaled_target), background=matrix.mean(axis=0)
    )


def by_hour(X):
    """Torna les posicions de les files de cada hora."""
    hours = X[HOUR].to_numpy()
    return [np.flatnonzero(hours == hour) for hour in range(HOURS_PER_DAY)]


def fit_rows(X, y, penalty, mixing):
    """Ajusta els 24 models, cada un amb les files de la seva hora i cap més."""
    return tuple(
        _fit_hour(X.iloc[rows], y.iloc[rows], penalty, mixing) for rows in by_hour(X)
    )


def largest_penalty(X, y, mixing):
    """Torna el càstig més petit que deixa a zero tots els coeficients de les 24 hores.

    És amb la barreja `mixing`. El de cada hora és el de scikit-learn, el màxim sobre
    les columnes de `|X'y|` dividit per les files i per la barreja, sobre la matriu i
    l'objectiu del model de l'hora centrats, perquè el model ajusta un terme constant.
    Un sol càstig serveix les 24 hores, i per deixar-les totes a zero ha de ser el més
    gran dels 24.
    """
    found = []
    for rows in by_hour(X):
        _, matrix, target = _prepared(X.iloc[rows], y.iloc[rows])
        product = (matrix - matrix.mean(axis=0)).T @ (target - target.mean())
        found.append(np.abs(product).max() / (len(target) * mixing))
    return max(found)


def predict_rows(model, X):
    """Torna la predicció de cada fila en EUR/MWh, del model de la seva hora."""
    predicted = np.empty(len(X))
    for fitted, rows in zip(model, by_hour(X), strict=True):
        if len(rows):
            transformed = fitted.model.predict(design(X.iloc[rows], fitted))
            predicted[rows] = fitted.target.back(transformed[:, np.newaxis]).ravel()
    return pd.Series(predicted, index=X.index, name=NAME)
