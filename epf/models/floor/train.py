"""El sòl de l'ordre de mèrit de Trebbien et al. (2023): un polinomi cúbic de la càrrega
residual a l'hora, la demanda menys l'eòlica i la fotovoltaica previstes, ajustat amb les
files d'entrenament de cada llavor i puntuat sobre les de prova. Es puntua i no s'explica.
"""
import time

import numpy as np

from epf import paths
from epf.contract import features
from epf.contract.metrics import scored
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

FLOOR = "floor"

# Grau del polinomi, el cúbic de Trebbien et al. (2023).
FLOOR_DEGREE = 3

# La família, com la planifica una passada: el grau, i cap cerca.
FAMILIES = {FLOOR: {"fixed": {"degree": FLOOR_DEGREE}, "placeholder": None, "kept": 0,
                    "searched_as": FLOOR}}


def _fitted(X, y, parts):
    """El cúbic de les files d'entrenament, i el que se'n desa per dibuixar-lo."""
    training, test = X.index.isin(parts.training), X.index.isin(parts.test)
    load = (X["demand"] - X["wind"] - X["solar"]).to_numpy()
    coefficients = np.polyfit(load[training], y[training].to_numpy(), FLOOR_DEGREE)
    return Fitted(np.polyval(coefficients, load[test]), y[test].to_numpy(),
                  {"coefficients": coefficients, "load_training": load[training],
                   "actual_training": y[training].to_numpy()})


def train(family, folder, configurations, explained_days):
    """La passada del sòl: per a cada llavor, el cúbic ajustat, puntuat i desat. Torna les
    parelles `[llavor, configuració]` que ha desat ara."""
    X, y = features.tables(*features.series()).hourly
    done = []
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        explained = parts.test if explained_days is None else parts.test[:explained_days]
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            start = time.time()
            fitted = _fitted(X, y, parts)
            scores = scored(fitted.predicted, fitted.actual)
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
            print(f"{family} llavor {seed} configuració {rank}: R² {scores['r2']:.3f}, "
                  f"MAE {scores['mae']:.2f} EUR/MWh, {time.time() - start:.1f} s",
                  flush=True)
    return done
