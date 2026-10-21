"""La sonda de datació de Trebbien et al. (2023), l'arbre amb un comptador de dies.

És una família pròpia, ajustada sobre la matriu per hora amb el comptador de més. Amb
una partició barrejada, una setmana retinguda queda entre setmanes d'entrenament, i un
model que sap situar una fila en el temps pot interpolar el nivell del període en lloc
de llegir-lo de l'entrada: la part de l'atribució que pren el comptador, i el que treu
de l'error, diuen quant ho fa. S'ajusta amb la millor configuració de cada seed de la
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
from epf.models.lightgbm.hyperparameters import DATE, DAY_COUNTER, NAME
from epf.models.lightgbm.train import FAMILIES as TREE
from epf.models.lightgbm.train import fitted
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La família, com la planifica una passada: els números de l'arbre, i la millor
# configuració de cada seed de la cerca de l'arbre.
FAMILIES = {"date_probe": TREE[NAME] | {"kept": 1, "searched_as": NAME}}


def train(family, folder, configurations, explained_days):
    """Ajusta, comprova, puntua i desa cada configuració del pla que encara no hi és.

    `configurations` és el pla de la família per seed, i `explained_days` els dies de
    prova que s'expliquen de cada seed, tots amb `None`.
    """
    rows, y = matrix.matrices(matrix.series()).hourly
    counter = (rows.index - pd.Timestamp(DEVELOPMENT[0])).days
    X = rows.assign(**{DAY_COUNTER: counter.to_numpy()})
    groups = np.array(
        [DATE if column == DAY_COUNTER else group(column) for column in X.columns]
    )
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
                # Sense interaccions: la lectura de la sonda és la part del comptador i
                # el seu error.
                best=False,
            )
            check(found, rank)
            scores = scored(found.predicted, found.actual)
            save_model(path, parts, explained, found, scores, rank)
