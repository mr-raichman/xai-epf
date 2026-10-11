"""Gradient boosting sobre la taula per hora: un sol model per a totes les hores.

LightGBM no fa sortida múltiple i TreeSHAP tampoc no l'admetria, així que la
família comparteix un model entre les 24 hores i rep l'hora com a columna. A
canvi, les interaccions de SHAP amb l'hora diuen com canvia un efecte al llarg
del dia.
"""
import numpy as np
import shap
from lightgbm import LGBMRegressor, early_stopping

from epf.matrix import calibration
from epf.models.boosting.boosting_hyperparameters import (LEAVES, PATIENCE, RATE, TREES,
                                                          WINDOW)
from epf.parameters import SEED
from epf.attribution.stored import ONE, PRICE_UNITS, Attribution

# Nom de la família i del seu únic pronòstic. El mòdul es diu `boosting` perquè és la
# tècnica; la família es diu per la implementació, que és el que es compara.
NAME = "lightgbm"


def lightgbm(trees=TREES, rate=RATE, leaves=LEAVES, seed=SEED):
    """Un regressor de LightGBM sense ajustar, i repetible.

    `deterministic` amb `force_row_wise` perquè dos ajustos de la mateixa finestra
    donin els mateixos arbres: sense això no els donen, i la predicció no ho
    destapa perquè segueix un sol camí per arbre mentre SHAP recorre tots els camins
    pesats per la cobertura. Mesurat el 2026-10-07 sobre la finestra de 1085 dies, tres
    ajustos: 1,21 s cada un i models diferents sense les dues banderes, 1,88 s i
    idèntics amb elles. No són números que es toquin per guanyar error, i per això no
    viuen al fitxer d'hiperparàmetres.
    """
    return LGBMRegressor(n_estimators=trees, learning_rate=rate, num_leaves=leaves,
                         random_state=seed, n_jobs=-1, verbose=-1,
                         deterministic=True, force_row_wise=True)


def fit_rows(X, y, validation=(), make=lightgbm, patience=PATIENCE):
    """Un model ajustat amb les files donades.

    Amb `validation`, parelles `(X, y)`, l'entrenament s'atura quan qualsevol d'elles
    porta `patience` arbres sense millorar, i el model es queda amb el millor arbre
    d'aquella: és l'aturada de Trebbien et al. (2023), que parteix la validació en
    quatre. `make` torna un regressor nou cada vegada, de manera que cap ajust no
    hereta res de l'anterior.
    """
    if not validation:
        return make().fit(X, y)
    return make().fit(X, y, eval_set=list(validation),
                      callbacks=[early_stopping(patience, verbose=False)])


def predict_rows(model, X):
    """El preu de cada fila."""
    return model.predict(X)


def attribute_rows(model, X):
    """Els valors de SHAP de cada fila, `(files, columnes)`, i la base.

    TreeSHAP al mode que LightGBM calcula ell mateix, i un sol model per a les 24
    hores, de manera que la base és la mateixa per a totes les files.
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X), float(np.ravel(explainer.expected_value)[0])


def fit(X, y, day, make=lightgbm, window=WINDOW):
    """El model ajustat només amb les hores dels dies anteriors a `day`."""
    return fit_rows(*calibration(X, y, day, window), make=make)


def predict(model, X, day):
    """Els 24 preus de `day`, en ordre d'hora, per nom de pronòstic."""
    return {NAME: predict_rows(model, X.loc[[day]])}


def attribute(model, X, day):
    """L'atribució de les 24 hores de `day`, del model que les ha predit.

    Les 24 files del dia donen `(hores, columnes)`, que es desa transposat, com la
    resta, en `(columna, hora)`.
    """
    rows = X.loc[[day]]
    values, base = attribute_rows(model, rows)
    return Attribution(ONE, list(X.columns), values.T[np.newaxis],
                       np.full((1, len(rows)), base), None, None, PRICE_UNITS)
