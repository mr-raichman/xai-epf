"""La passada de la xarxa neuronal.

Es puntua i no s'explica.
"""

from epf import paths
from epf.contract import matrix
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.dnn import network
from epf.models.dnn.hyperparameters import FIXED
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# Nom del model i de la seva família.
NAME = "dnn"

# La família, com la planifica una passada: tots els números fixos que llegeixen els
# seus trials i els seus ajustos, i la millor configuració de cada seed de la seva
# pròpia cerca.
FAMILIES = {
    NAME: {"fixed": FIXED, "kept": 1, "searched_as": NAME, "ljung_box_tested": True}
}


def _fitted(X, y, parts, seed, numbers):
    """Ajusta la xarxa d'una configuració i en prediu les setmanes de prova."""
    training, validation, test = (X.index.isin(days) for days in parts)
    fitted = network.fit_rows(
        (X[training], y[training]),
        (X[validation], y[validation]),
        numbers,
        seed,
    )
    return Fitted(
        network.predict_rows(fitted, X[test]).ravel(),
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
