"""La sonda de datació de Trebbien et al. (2023), l'arbre amb un comptador de dies.

És una família pròpia, ajustada sobre la matriu per hora amb el comptador de més. Amb
una partició barrejada, una setmana retinguda queda entre setmanes d'entrenament, i un
model que sap situar una fila en el temps pot interpolar el nivell del període en lloc
de llegir-lo de l'entrada: la part de l'atribució que pren el comptador, i el que treu
de l'error, diuen quant ho fa. S'ajusta amb la millor configuració de cada llavor de la
cerca de l'arbre, de manera que només el comptador la separa de l'arbre amb què es
compara.
"""

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import matrix
from epf.contract.explanation import check, group
from epf.contract.metrics import scored
from epf.contract.record import save_model
from epf.contract.split import split
from epf.models.lightgbm.hyperparameters import NAME
from epf.models.lightgbm.train import FAMILIES as TREE
from epf.models.lightgbm.train import fitted
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La columna del comptador, els dies des del primer del període, i el seu grup a la
# importància per grups, que no és cap sèrie ni el calendari.
DAY_COUNTER = "day_counter"
DATE = "date"

# La família, com la planifica una passada: els números de l'arbre, i la millor
# configuració de cada llavor de la cerca de l'arbre.
FAMILIES = {"date_probe": TREE[NAME] | {"kept": 1, "searched_as": NAME}}


def train(family, folder, configurations, explained_days):
    """Ajusta, comprova, puntua i desa cada configuració del pla que encara no hi és.

    La matriu és la per hora amb el comptador. `configurations` és el pla de la família
    per llavor, i `explained_days` els dies de prova que s'expliquen de cada llavor,
    tots amb `None`. Torna les parelles `[llavor, configuració]` que ha desat ara.
    """
    rows, y = matrix.matrices(matrix.series()).hourly
    counter = (rows.index - pd.Timestamp(DEVELOPMENT[0])).days
    X = rows.assign(**{DAY_COUNTER: counter.to_numpy()})
    groups = np.array(
        [DATE if column == DAY_COUNTER else group(column) for column in X.columns]
    )
    done = []
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        explained = (
            parts.test if explained_days is None else parts.test[:explained_days]
        )
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            found = fitted(
                X,
                y,
                groups,
                parts,
                explained,
                seed,
                configuration["numbers"],
                best=rank == 0,
            )
            check(found, rank)
            scores = scored(found.predicted, found.actual)
            save_model(path, parts, explained, found, scores, rank)
            done.append([seed, rank])
    return done
