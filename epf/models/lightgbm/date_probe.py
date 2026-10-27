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
from epf.contract.explanation import DATE, group
from epf.contract.record import save_model
from epf.contract.split import split
from epf.models.lightgbm import hyperparameters
from epf.models.lightgbm.hyperparameters import DAY_COUNTER
from epf.models.lightgbm.train import FAMILIES as TREE
from epf.models.lightgbm.train import fitted
from epf.parameters import DEVELOPMENT

# Nom de la família.
NAME = "date_probe"

# La família, com la planifica una passada: els números de l'arbre, i la millor
# configuració de cada seed de la cerca de l'arbre; i la columna del seu comptador.
FAMILIES = {
    NAME: TREE[hyperparameters.NAME]
    | {"kept": 1, "searched_as": hyperparameters.NAME, "counter": DAY_COUNTER}
}


def train(family, folder, seed, configuration, explained_days):
    """Ajusta, comprova, puntua i desa una configuració del pla d'una seed.

    `explained_days` són els dies de prova que s'expliquen, tots amb `None`.
    """
    built = matrix.matrices(matrix.series())
    rows, y = built.hourly
    counter = (rows.index - pd.Timestamp(DEVELOPMENT[0])).days
    X = rows.assign(**{DAY_COUNTER: counter.to_numpy()})
    groups = np.array(
        [DATE if column == DAY_COUNTER else group(column) for column in X.columns]
    )
    parts = split(seed, DEVELOPMENT, built.deleted)
    explained = parts.test if explained_days is None else parts.test[:explained_days]
    rank = configuration["rank"]
    found = fitted(
        X,
        y,
        groups,
        parts,
        built.deleted,
        explained,
        seed,
        configuration["numbers"],
        with_interactions=False,
    )
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, explained, found, rank)
