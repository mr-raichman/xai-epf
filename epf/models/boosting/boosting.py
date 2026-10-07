"""Gradient boosting sobre la taula per hora: un sol model per a totes les hores.

LightGBM no fa sortida múltiple i TreeSHAP tampoc no l'admetria, així que la
família comparteix un model entre les 24 hores i rep l'hora com a columna. A
canvi, les interaccions de SHAP amb l'hora diuen com canvia un efecte al llarg
del dia.
"""
import numpy as np
import shap
from lightgbm import LGBMRegressor

from epf.matrix import calibration
from epf.models.boosting.boosting_hyperparameters import LEAVES, RATE, TREES, WINDOW
from epf.parameters import SEED
from epf.models.attribution import ONE, PRICE_UNITS, Attribution

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


def fit(X, y, day, make=lightgbm, window=WINDOW):
    """El model ajustat només amb les hores dels dies anteriors a `day`.

    `make` torna un regressor nou cada vegada, de manera que cap dia no hereta res
    de l'anterior.
    """
    features, target = calibration(X, y, day, window)
    return make().fit(features, target)


def predict(model, X, day):
    """Els 24 preus de `day`, en ordre d'hora."""
    return model.predict(X.loc[[day]])


def attribute(model, X, day):
    """L'atribució de les 24 hores de `day`, del model que les ha predit.

    TreeSHAP torna (hores, columnes) per a les 24 files del dia; es desa transposat,
    com la resta, en (columna, hora).
    """
    rows = X.loc[[day]]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(rows)
    # Un sol model per a les 24 hores, de manera que la base és la mateixa a totes.
    base = np.full((1, len(rows)), float(np.ravel(explainer.expected_value)[0]))
    return Attribution(ONE, list(X.columns), values.T[np.newaxis], base, None,
                       PRICE_UNITS)
