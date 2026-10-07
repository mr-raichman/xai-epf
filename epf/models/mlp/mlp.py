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

from epf.matrix import calibration
from epf.models.mlp.mlp_hyperparameters import (ALPHA, HIDDEN, LEARNING_RATE,
                                                MAX_ITER, MEMBERS, NO_CHANGE,
                                                VALIDATION_FRACTION, WINDOW)
from epf.parameters import SEED


def fit(X, y, day, window=WINDOW, hidden=HIDDEN, members=MEMBERS, seed=SEED,
        alpha=ALPHA, learning_rate=LEARNING_RATE,
        validation_fraction=VALIDATION_FRACTION, no_change=NO_CHANGE,
        max_iter=MAX_ITER):
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
        MLPRegressor(hidden_layer_sizes=tuple(hidden), alpha=alpha,
                     learning_rate_init=learning_rate, max_iter=max_iter,
                     early_stopping=True, validation_fraction=validation_fraction,
                     n_iter_no_change=no_change, random_state=seed + member)
        .fit(scaled, scaled_target)
        for member in range(members)]
    return networks, inputs, outputs


def predict(model, X, day):
    """Els 24 preus de `day`, la mitjana de les xarxes del model."""
    networks, inputs, outputs = model
    row = inputs.transform(X.loc[[day]])
    return np.mean([outputs.inverse_transform(network.predict(row))[0]
                    for network in networks], axis=0)
