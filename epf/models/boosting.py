"""Gradient boosting sobre la taula per hora: un sol model per a totes les hores.

TreeSHAP no admet ni les fulles vectorials d'XGBoost ni cap sortida múltiple de
LightGBM, així que la família comparteix un model entre les 24 hores i rep l'hora
com a columna. A canvi, les interaccions de SHAP amb l'hora diuen com canvia un
efecte al llarg del dia.

Les dues llibreries es mesuren des del mateix punt de partida: els seus valors per
defecte no ho són, i una comparació entre elles amb els seus propis valors
compararia els valors i no les llibreries.
"""
import numpy as np
import shap
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from epf.features.matrix import calibration
from epf.models.attribution import ONE, PRICE_UNITS, Attribution

# Punt de partida, per ajustar sobre el període de validació. Els arbres són els
# mateixos que els del bosc, perquè una diferència entre les dues famílies d'arbres
# no sigui una diferència de pressupost; una profunditat de 5 són 32 fulles com a
# molt, equivalent a les 31 de LightGBM.
TREES = 500
LEARNING_RATE = 0.05
LEAVES = 31
MAX_DEPTH = 5
SEED = 0


def lightgbm(trees=TREES, rate=LEARNING_RATE, leaves=LEAVES, seed=SEED):
    """Un regressor de LightGBM sense ajustar."""
    return LGBMRegressor(n_estimators=trees, learning_rate=rate, num_leaves=leaves,
                         random_state=seed, n_jobs=-1, verbose=-1)


def xgboost(trees=TREES, rate=LEARNING_RATE, depth=MAX_DEPTH, seed=SEED):
    """Un regressor d'XGBoost sense ajustar."""
    return XGBRegressor(n_estimators=trees, learning_rate=rate, max_depth=depth,
                        random_state=seed, n_jobs=-1)


def fit(make, X, y, day, window=None):
    """El model ajustat només amb les hores dels dies anteriors a `day`.

    `make` torna un regressor nou cada vegada, de manera que cap dia no hereta res
    de l'anterior.
    """
    features, target = calibration(X, y, day, window)
    return make().fit(features, target)


def forecast(make, X, y, day, window=None):
    """Els 24 preus de `day`, en ordre d'hora."""
    return fit(make, X, y, day, window).predict(X.loc[[day]])


def explain(make, X, y, day, window=None):
    """Els 24 preus de `day` i l'atribució del mateix model que els ha predit.

    TreeSHAP torna (hores, columnes) per a les 24 files del dia; es desa transposat,
    com la resta, en (columna, hora).
    """
    fitted = fit(make, X, y, day, window)
    rows = X.loc[[day]]
    values = shap.TreeExplainer(fitted).shap_values(rows)
    return (fitted.predict(rows),
            Attribution(ONE, list(X.columns), values.T[np.newaxis], PRICE_UNITS))
