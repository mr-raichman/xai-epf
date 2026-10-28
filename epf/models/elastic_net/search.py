"""La cerca de l'elastic net, sobre la graella d'Uniejewski et al. (2016)."""

import numpy as np

from epf.contract.metrics import mae
from epf.models.elastic_net import regression
from epf.models.elastic_net.hyperparameters import NAME
from epf.models.elastic_net.train import FAMILIES

# La matriu de la cerca, la per hora, com la de la passada.
MATRIX = "hourly"

# Les barreges d'Uniejewski et al. (2016), els càstigs de cada una, i la raó entre el
# més petit i el més gran, el `LambdaRatio` per defecte de `lasso.m`.
MIXINGS = (0.25, 0.5, 0.75)
STEPS = 34
RATIO = 1e-4

# La graella: cada barreja amb cada pas de la seva seqüència.
GRID = {"mixing": MIXINGS, "step": tuple(range(STEPS))}

# Com es cerca, que el study desa i compara amb el d'ara: les barreges, els passos i la
# raó dels càstigs, i els números fixos de la família.
SPACE = {"mixings": MIXINGS, "steps": STEPS, "ratio": RATIO}
FIXED = FAMILIES[NAME]["fixed"]


def penalties(X, y):
    """Torna els `STEPS` càstigs de cada barreja, sobre les files donades.

    Van del que deixa a zero tots els coeficients de les 24 hores cap avall, en sèrie
    geomètrica fins a `RATIO` d'ell.
    """
    found = {}
    for mixing in MIXINGS:
        largest = regression.largest_penalty(X, y, mixing)
        found[mixing] = np.geomspace(largest, largest * RATIO, STEPS)
    return found


def drawn(family, optuna_trial):
    """Torna el punt de la graella que proposa un trial: la barreja i el pas."""
    return tuple(
        optuna_trial.suggest_categorical(name, values) for name, values in GRID.items()
    )


def trial(family, drawn, train, validate, seed):
    """Ajusta el punt d'un trial, `drawn`, i el puntua pel MAE de la validació.

    El càstig del seu pas es calcula sobre les files d'entrenament, `train`. Torna el
    càstig i la barreja, i la puntuació.
    """
    (X, y), (X_validate, y_validate) = train, validate
    mixing, step = drawn
    penalty = float(penalties(X, y)[mixing][step])
    model = regression.fit_rows(X, y, penalty, mixing)
    return (
        {"penalty": penalty, "mixing": mixing},
        mae(regression.predict_rows(model, X_validate), y_validate),
    )
