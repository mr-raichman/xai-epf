"""La xarxa elàstica d'Uniejewski et al. (2016): un model lineal per hora, explicat per
l'atribució exacta d'un model lineal.

Cada hora té el seu model, ajustat només amb les files de la seva hora de la taula per
hora, que porten les tres previsions a aquella hora, com l'article hi porta la previsió
de càrrega. Preus i entrades passen per la transformació de LEAR, que és d'on la pren:
la logarítmica de l'article no admet preus a zero o per sota. La penalització és una
per a les 24 hores, com a l'article, i ni ella ni la barreja tenen valor per defecte,
perquè les tria una cerca.

Tot és sobre files donades: la part B hi ajusta les files d'entrenament d'una llavor,
i el camí endavant les d'una finestra, amb les tres funcions del final.
"""
import collections

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from epf.attribution.stored import ONE, TRANSFORMED_UNITS, Attribution
from epf.market import HOURS_PER_DAY
from epf.matrix import DAY_OF_WEEK_COLUMNS, HOUR, calibration
from epf.models.linear.linear_hyperparameters import MAX_ITER, NAME, WINDOW, searched
from epf.models.scaling import Scaler

# El model d'una hora i el que cal per llegir-lo. `columns` són les de la matriu del
# model en el seu ordre, les escalades i després els indicadors del dia de la setmana,
# sense cap de les que no es mouen a les files de l'hora. `inputs` escala les entrades,
# `target` l'objectiu i és qui torna a EUR/MWh, i `background` és la mitjana de cada
# columna de la matriu, d'on surten les contribucions.
Hour = collections.namedtuple("Hour", "model columns inputs target background")

# L'atribució de cada fila, a l'espai transformat: les contribucions per columna, la
# base de l'hora de la fila i la predicció de la fila al mateix espai, que és la que
# base més contribucions ha de tornar.
Explanation = collections.namedtuple("Explanation", "values base transformed")


def _design(rows, fitted):
    """Les files com la matriu del model d'una hora: entrades transformades i indicadors."""
    scaled = len(fitted.inputs.median)
    values = rows[fitted.columns].to_numpy(dtype=float)
    return np.hstack([fitted.inputs.forward(values[:, :scaled]), values[:, scaled:]])


def _prepared(rows, target):
    """Les files d'una hora com les veu el seu model: el que cal per llegir-lo, encara
    sense model, la matriu i l'objectiu transformat.

    Una columna que no es mou a les files de l'hora no hi entra, que és la previsió solar
    de nit. La que es mou sense MAD s'escala per la seva desviació típica, que és la
    solar dels vespres d'estiu: la transformació de LEAR la refusaria.
    """
    moving = [column for column in rows.columns
              if column != HOUR and rows[column].nunique() > 1]
    scaled = [column for column in moving if column not in DAY_OF_WEEK_COLUMNS]
    dummies = [column for column in moving if column in DAY_OF_WEEK_COLUMNS]
    inputs = Scaler(rows[scaled].to_numpy(dtype=float), std_where_flat=True)
    output = Scaler(target.to_numpy(dtype=float)[:, np.newaxis])
    fitted = Hour(None, scaled + dummies, inputs, output, None)
    scaled_target = output.forward(target.to_numpy(dtype=float)[:, np.newaxis]).ravel()
    return fitted, _design(rows, fitted), scaled_target


def _fit_hour(rows, target, penalty, mixing, max_iter):
    """El model d'una hora, amb les seves files i prou."""
    fitted, design, scaled_target = _prepared(rows, target)
    model = ElasticNet(alpha=penalty, l1_ratio=mixing, max_iter=max_iter)
    return fitted._replace(model=model.fit(design, scaled_target),
                           background=design.mean(axis=0))


def _by_hour(X):
    """La posició de les files de cada hora."""
    hours = X[HOUR].to_numpy()
    return [np.flatnonzero(hours == hour) for hour in range(HOURS_PER_DAY)]


def fit_rows(X, y, penalty, mixing, max_iter=MAX_ITER):
    """Els 24 models, cada un ajustat amb les files de la seva hora i cap més."""
    return tuple(_fit_hour(X.iloc[rows], y.iloc[rows], penalty, mixing, max_iter)
                 for rows in _by_hour(X))


def largest_penalty(X, y, mixing):
    """El càstig més petit que deixa a zero tots els coeficients de les 24 hores, amb
    aquesta barreja.

    El de cada hora és el de scikit-learn, el màxim sobre les columnes de `|X'y|` dividit
    per les files i per la barreja, sobre la matriu i l'objectiu del model de l'hora
    centrats, perquè el model ajusta un terme constant. Un sol càstig serveix les 24
    hores, i per deixar-les totes a zero ha de ser el més gran dels 24.
    """
    found = []
    for rows in _by_hour(X):
        _, design, target = _prepared(X.iloc[rows], y.iloc[rows])
        product = (design - design.mean(axis=0)).T @ (target - target.mean())
        found.append(np.abs(product).max() / (len(target) * mixing))
    return max(found)


def predict_rows(model, X):
    """El preu de cada fila en EUR/MWh, cada una amb el model de la seva hora."""
    prices = np.empty(len(X))
    for fitted, rows in zip(model, _by_hour(X)):
        if len(rows):
            transformed = fitted.model.predict(_design(X.iloc[rows], fitted))
            prices[rows] = fitted.target.back(transformed[:, np.newaxis]).ravel()
    return pd.Series(prices, index=X.index, name=NAME)


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
    for fitted, rows in zip(model, _by_hour(X)):
        if not len(rows):
            continue
        design = _design(X.iloc[rows], fitted)
        coefficients = fitted.model.coef_
        positions = [columns.index(column) for column in fitted.columns]
        values[np.ix_(rows, positions)] = coefficients * (design - fitted.background)
        base[rows] = fitted.model.intercept_ + coefficients @ fitted.background
        transformed[rows] = fitted.model.predict(design)
    return Explanation(pd.DataFrame(values, index=X.index, columns=columns),
                       pd.Series(base, index=X.index), pd.Series(transformed, index=X.index))


def fit(X, y, day, window=WINDOW, numbers=None):
    """Els 24 models ajustats només amb les hores dels dies anteriors a `day`.

    El càstig i la barreja són els de `searched()` si no se'n passen, els mateixos cada
    dia, com Uniejewski et al. (2016) mantenen el seu durant tot el període de prova.
    """
    numbers = searched() if numbers is None else numbers
    return fit_rows(*calibration(X, y, day, window), numbers["penalty"],
                    numbers["mixing"])


def predict(model, X, day):
    """Els 24 preus de `day`, en ordre d'hora, per nom de pronòstic."""
    return {NAME: predict_rows(model, X.loc[[day]]).to_numpy()}


def attribute(model, X, day):
    """L'atribució de les 24 hores de `day`, del model que les ha predit.

    És a l'espai transformat, i es desa amb el que la deixa comprovar: la predicció de
    cada hora en EUR/MWh, que és la del CSV, i la mateixa a l'espai transformat, que és
    la que la base i les contribucions han de tornar. Les files del dia donen
    `(hores, columnes)`, que es desa transposat, com la resta, en `(columna, hora)`.
    """
    rows = X.loc[[day]]
    explained = attribute_rows(model, rows)
    return Attribution(ONE, list(explained.values.columns),
                       explained.values.to_numpy().T[np.newaxis],
                       explained.base.to_numpy()[np.newaxis],
                       predict_rows(model, rows).to_numpy()[np.newaxis],
                       explained.transformed.to_numpy()[np.newaxis], TRANSFORMED_UNITS)
