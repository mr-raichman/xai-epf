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
    """Un regressor de LightGBM sense ajustar."""
    return LGBMRegressor(n_estimators=trees, learning_rate=rate, num_leaves=leaves,
                         random_state=seed, n_jobs=-1, verbose=-1)


def fit(X, y, day, make=lightgbm, window=WINDOW):
    """El model ajustat només amb les hores dels dies anteriors a `day`.

    `make` torna un regressor nou cada vegada, de manera que cap dia no hereta res
    de l'anterior.
    """
    features, target = calibration(X, y, day, window)
    return make().fit(features, target)


def forecast(X, y, day, make=lightgbm, window=WINDOW):
    """Els 24 preus de `day`, en ordre d'hora."""
    return fit(X, y, day, make, window).predict(X.loc[[day]])


def explain(X, y, day, make=lightgbm, window=WINDOW):
    """Els 24 preus de `day` i l'atribució del mateix model que els ha predit.

    TreeSHAP torna (hores, columnes) per a les 24 files del dia; es desa transposat,
    com la resta, en (columna, hora).
    """
    fitted = fit(X, y, day, make, window)
    rows = X.loc[[day]]
    explainer = shap.TreeExplainer(fitted)
    values = explainer.shap_values(rows)
    # Un sol model per a les 24 hores, de manera que la base és la mateixa a totes.
    base = np.full((1, len(rows)), float(np.ravel(explainer.expected_value)[0]))
    return (fitted.predict(rows),
            Attribution(ONE, list(X.columns), values.T[np.newaxis], base, None,
                        PRICE_UNITS))
