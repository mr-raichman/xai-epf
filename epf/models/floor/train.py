"""El sòl de l'ordre de mèrit de Trebbien et al. (2023), i la seva passada.

Un polinomi cúbic de la càrrega residual prevista a l'hora. Es puntua i no s'explica.
"""

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import matrix
from epf.contract.record import Fitted, save_model

# Nom del model i de la seva família.
NAME = "floor"

# Grau del polinomi, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

# Les columnes que llegeix de la matriu per hora, sense les quals no s'ajusta.
READS = ("demand_fcst", "wind_fcst", "solar_fcst")

# La família, com la planifica una passada: el grau, cap cerca, i el que llegeix.
FAMILIES = {
    NAME: {
        "fixed": {"degree": FLOOR_DEGREE},
        "kept": 0,
        "searched_as": NAME,
        "ljung_box_tested": True,
        "interpreted": False,
        "benchmark": True,
        "reads": READS,
    }
}


def _fitted(X, y, parts):
    """Ajusta el cúbic amb les files d'entrenament i en prediu les de prova.

    Torna també els coeficients i les files d'entrenament, per dibuixar-ne el cúbic.
    """
    training, test = X.index.isin(parts.training), X.index.isin(parts.test)
    demand, wind, solar = READS
    load = (X[demand] - X[wind] - X[solar]).to_numpy()
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


def train(family, folder, seed, configuration, explained_days, built):
    """Ajusta, puntua i desa una configuració del pla d'una seed.

    `explained_days` no hi fa res, perquè la família no explica res, i `built` són les
    matrius de la passada.
    """
    X, _ = built.hourly
    # El preu de cada fila, que el sòl prediu també on les altres famílies prediuen la
    # distància a la referència.
    y = pd.Series(built.actual.to_numpy().ravel(), index=X.index)
    parts = matrix.partition(built, seed)
    rank = configuration["rank"]
    fitted = _fitted(X, y, parts)
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, None, fitted, rank)
