"""La passada de la xarxa de punt de Marcjasz et al. (2023).

Es puntua i no s'explica.
"""

import keras

from epf import paths
from epf.contract import matrix
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.nn import network
from epf.models.nn.hyperparameters import FIXED
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La sortida de la xarxa de punt de l'article: els 24 preus tal com surten, i l'error
# absolut, la pèrdua `'mae'` del seu codi. No és cap distribució, però la xarxa la
# construeix i l'ajusta igual que les altres dues.
POINT = network.Distribution(
    names=("price",),
    loss=keras.losses.mean_absolute_error,
    flat=lambda raw: raw,
    point=lambda parameters: parameters[..., 0],
    quantiles=None,
)

# Nom del model i de la seva família.
NAME = "dnn"

# La família, com la planifica una passada: tots els números fixos que llegeixen els
# seus trials i els seus ajustos, i la millor configuració de cada seed de la seva
# pròpia cerca.
FAMILIES = {
    NAME: {"fixed": FIXED, "kept": 1, "searched_as": NAME, "residuals_checked": True}
}


def _fitted(X, y, parts, seed, numbers):
    """Ajusta la xarxa d'una configuració i en prediu les setmanes de prova."""
    training, validation, test = (X.index.isin(days) for days in parts)
    fitted = network.fit_rows(
        POINT,
        (X[training], y[training]),
        (X[validation], y[validation]),
        numbers,
        seed,
    )
    return Fitted(
        network.predict_rows(POINT, fitted, X[test]).ravel(),
        y[test].to_numpy().ravel(),
        {},
    )


def train(family, folder, seed, configuration, explained_days):
    """Ajusta, puntua i desa una configuració del pla d'una seed.

    `explained_days` no hi fa res, perquè la família no explica res.
    """
    X, y = matrix.matrices(matrix.series()).daily
    parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
    rank = configuration["rank"]
    fitted = _fitted(X, y, parts, seed, configuration["numbers"])
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, None, fitted, rank)
