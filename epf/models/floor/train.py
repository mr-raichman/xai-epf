"""El sòl de l'ordre de mèrit de Trebbien et al. (2023), i la seva passada.

Un polinomi cúbic de la càrrega residual prevista a l'hora. Es puntua i no s'explica.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.parameters import DEVELOPMENT

# Nom del model i de la seva família.
NAME = "floor"

# Grau del polinomi, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

# La família, com la planifica una passada: el grau, i cap cerca.
FAMILIES = {
    NAME: {
        "fixed": {"degree": FLOOR_DEGREE},
        "kept": 0,
        "searched_as": NAME,
        "ljung_box_tested": True,
    }
}


def _fitted(X, y, parts):
    """Ajusta el cúbic amb les files d'entrenament i en prediu les de prova.

    Torna també els coeficients i les files d'entrenament, que dibuixa la figura.
    """
    training, test = X.index.isin(parts.training), X.index.isin(parts.test)
    load = (X["demand_fcst"] - X["wind_fcst"] - X["solar_fcst"]).to_numpy()
    coefficients = np.polyfit(load[training], y[training].to_numpy(), FLOOR_DEGREE)
    return Fitted(
        np.polyval(coefficients, load[test]),
        y[test].to_numpy(),
        {
            "coefficients": coefficients,
            "load_training": load[training],
            "actual_training": y[training].to_numpy(),
        },
    )


def train(family, folder, seed, configuration, explained_days):
    """Ajusta, puntua i desa una configuració del pla d'una seed.

    `explained_days` no hi fa res, perquè la família no explica res.
    """
    built = matrix.matrices(matrix.series())
    X, y = built.hourly
    parts = split(seed, DEVELOPMENT, built.deleted)
    rank = configuration["rank"]
    fitted = _fitted(X, y, parts)
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, None, fitted, rank)
