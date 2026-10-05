"""Xarxa connexa sobre la matriu diària: una fila per dia i les 24 hores de sortida.

Dues capes amagades, Adam i aturada anticipada. És l'estructura de la xarxa de
referència de Lago et al. (2021), que en compta quatre incloent-hi l'entrada i la
sortida, però **no és la seva xarxa**: scikit-learn no té ni dropout, ni
normalització per lots, ni regularització L1, i ajusta l'error quadràtic allà on
l'article ajusta l'absolut.

La predicció és la mitjana de quatre xarxes que només es diferencien en la llavor,
com fa l'article: una xarxa és sensible a la inicialització, i mitjanar-les en treu
la variància en lloc de deixar-la sense mesurar.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from epf.features.matrix import calibration

# Punt de partida, que s'ajusta sobre el període de validació.
HIDDEN = (128, 128)
SEED = 0

# Xarxes que es mitjanen, el nombre de l'article.
MEMBERS = 4

# Valors per defecte de scikit-learn, que també són punt de partida i s'ajusten
# sobre la validació.
ALPHA = 1e-4
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1
NO_CHANGE = 10

# Sostre d'èpoques. Mesurat sobre sis finestres reals de 1085 i 1449 dies i dues
# llavors: l'aturada anticipada salta entre l'època 33 i la 126, de manera que 200
# no talla cap ajust. Un ajust tallat a mitges no es queixa.
MAX_ITER = 200


def fit(X, y, day, window=None, hidden=HIDDEN, members=MEMBERS, seed=SEED):
    """Les xarxes d'un dia i els escaladors que les alimenten, ajustats només amb
    dies anteriors.

    Els escaladors surten de la finestra de calibratge i mai del dia que es prediu,
    i les xarxes se'ls reparteixen: entre elles només canvia la llavor.
    """
    features, target = calibration(X, y, day, window)
    inputs = StandardScaler().fit(features)
    outputs = StandardScaler().fit(target)
    scaled, scaled_target = inputs.transform(features), outputs.transform(target)
    networks = [
        MLPRegressor(hidden_layer_sizes=tuple(hidden), alpha=ALPHA,
                     learning_rate_init=LEARNING_RATE, max_iter=MAX_ITER,
                     early_stopping=True, validation_fraction=VALIDATION_FRACTION,
                     n_iter_no_change=NO_CHANGE, random_state=seed + member)
        .fit(scaled, scaled_target)
        for member in range(members)]
    return networks, inputs, outputs


def forecast(X, y, day, window=None, **parameters):
    """Els 24 preus de `day`, la mitjana de les xarxes ajustades només amb dies
    anteriors."""
    networks, inputs, outputs = fit(X, y, day, window, **parameters)
    row = inputs.transform(X.loc[[day]])
    return np.mean([outputs.inverse_transform(network.predict(row))[0]
                    for network in networks], axis=0)
