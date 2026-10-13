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
from epf.models.boosting.boosting_hyperparameters import (MIN_CHILD_WEIGHT, NAME,
                                                          PATIENCE, SUBSAMPLE_FREQ, WINDOW,
                                                          searched)
from epf.parameters import SEED
from epf.attribution.stored import ONE, PRICE_UNITS, Attribution


def lightgbm(point, trees, seed=SEED):
    """Un regressor de LightGBM sense ajustar, i repetible, d'un punt de la graella i amb
    `trees` arbres com a molt.

    La llavor fixa quines files pren cada arbre quan el punt en pren menys de totes.

    `deterministic` amb `force_row_wise` perquè dos ajustos de la mateixa finestra
    donin els mateixos arbres: sense això no els donen, i la predicció no ho
    destapa perquè segueix un sol camí per arbre mentre SHAP recorre tots els camins
    pesats per la cobertura. Mesurat el 2026-10-07 sobre la finestra de 1085 dies, tres
    ajustos: 1,21 s cada un i models diferents sense les dues banderes, 1,88 s i
    idèntics amb elles. No són números que es toquin per guanyar error, i per això no
    viuen al fitxer d'hiperparàmetres.
    """
    return LGBMRegressor(n_estimators=trees, learning_rate=point["learning_rate"],
                         num_leaves=point["num_leaves"], max_depth=point["max_depth"],
                         subsample=point["subsample"], subsample_freq=SUBSAMPLE_FREQ,
                         min_child_samples=point["min_child_samples"],
                         min_child_weight=MIN_CHILD_WEIGHT, random_state=seed, n_jobs=-1,
                         verbose=-1, deterministic=True, force_row_wise=True)


def fit_rows(X, y, point, trees, validation=(), patience=PATIENCE):
    """Un model ajustat amb les files donades, d'un punt de la graella.

    Sense `validation` fa `trees` arbres, ni un més ni un menys. Amb `validation`,
    parelles `(X, y)`, `trees` és el sostre, i l'entrenament s'atura quan qualsevol
    d'elles porta `patience` arbres sense millorar, i el model es queda amb el millor
    arbre d'aquella: amb una sola parella és l'aturada de la cerca, i amb quatre la de
    Trebbien et al. (2023), que parteix la validació en quatre. Cada crida fa un
    regressor nou, de manera que cap ajust no hereta res de l'anterior.
    """
    model = lightgbm(point, trees)
    if not validation:
        return model.fit(X, y)
    parts, targets = zip(*validation)
    return model.fit(X, y, eval_X=parts, eval_y=targets,
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


def interaction_rows(model, X):
    """Els valors d'interacció de SHAP de cada fila, `(files, columnes, columnes)`."""
    return shap.TreeExplainer(model).shap_interaction_values(X)


def fit(X, y, day, point=None, window=WINDOW):
    """El model ajustat només amb les hores dels dies anteriors a `day`.

    El punt per defecte és el de `searched()`, amb el nombre d'arbres en què es va aturar
    l'ajust de la cerca, que cada dia es fan sencers i sense aturar-se: és el que fa el
    codi del grup de Trebbien et al. (2023), que torna a ajustar el punt triat amb el
    millor nombre d'arbres.
    """
    point = searched() if point is None else point
    return fit_rows(*calibration(X, y, day, window), point, point["trees"])


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
