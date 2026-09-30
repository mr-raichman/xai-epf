"""Bosc aleatori sobre la matriu diària: una fila per dia i les 24 hores de sortida.

Cada divisió redueix l'error de les 24 hores alhora i cada fulla guarda una corba
diària sencera, que és el que el fa multivariant de debò.
"""
from sklearn.ensemble import RandomForestRegressor

from epf.features.matrix import calibration

# Punt de partida, per ajustar sobre el període de validació. Una tercera part de
# les columnes a cada divisió és la recomanació habitual en regressió; amb totes,
# els arbres queden massa correlacionats i el bosc deixa de ser aleatori.
TREES = 500
FEATURES_PER_SPLIT = 1 / 3
SEED = 0


def fit(X, y, day, window=None, trees=TREES, features_per_split=FEATURES_PER_SPLIT, seed=SEED):
    """Bosc ajustat només amb els dies anteriors a `day`."""
    features, target = calibration(X, y, day, window)
    forest = RandomForestRegressor(n_estimators=trees, max_features=features_per_split,
                                   random_state=seed, n_jobs=-1)
    return forest.fit(features, target)


def forecast(X, y, day, window=None, **parameters):
    """Els 24 preus de `day`, del bosc ajustat només amb dies anteriors."""
    return fit(X, y, day, window, **parameters).predict(X.loc[[day]])[0]
