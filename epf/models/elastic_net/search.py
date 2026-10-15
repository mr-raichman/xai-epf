"""La cerca de la xarxa elàstica: la graella d'Uniejewski et al. (2016) i com s'hi prova
cada punt.

Les tres barreges que l'article fixa a mà, i per a cada una els 34 càstigs de la
seqüència que fa `lasso.m`, l'eina amb què l'article les ajusta: geomètrica, del càstig
més petit que deixa a zero tots els coeficients fins a `RATIO` d'ell, el valor per
defecte de l'eina, que l'article no canvia. Un sol càstig serveix les 24 hores, com a
l'article, de manera que el més gran de cada seqüència és el més gran de les 24. La
seqüència es calcula sobre les files d'entrenament.
"""
import numpy as np

from epf.contract.metrics import mae
from epf.models.elastic_net import regression

# La matriu de la cerca, la per hora, com la de la passada.
MATRIX = "hourly"

# Les barreges d'Uniejewski et al. (2016), els càstigs de cada una, i la raó entre el més
# petit i el més gran, el `LambdaRatio` per defecte de `lasso.m`.
MIXINGS = (0.25, 0.5, 0.75)
STEPS = 34
RATIO = 1e-4

# La graella: cada barreja amb cada pas de la seva seqüència, 102 punts.
GRID = {"mixing": MIXINGS, "step": tuple(range(STEPS))}


def penalties(X, y):
    """Per a cada barreja, els seus `STEPS` càstigs, del que deixa a zero tots els
    coeficients de les 24 hores cap avall, sobre les files donades."""
    found = {}
    for mixing in MIXINGS:
        largest = regression.largest_penalty(X, y, mixing)
        found[mixing] = np.geomspace(largest, largest * RATIO, STEPS)
    return found


def space(optuna_trial):
    """El punt de la graella d'una prova: la barreja i el pas de la seva seqüència."""
    return tuple(optuna_trial.suggest_categorical(name, values)
                 for name, values in GRID.items())


def trial(family, optuna_trial, train, validate, seed):
    """Un punt de la graella, amb el càstig del seu pas calculat sobre les files
    d'entrenament, `(X, y)`, i puntuat pel MAE de les de validació."""
    (X, y), (X_validate, y_validate) = train, validate
    mixing, step = space(optuna_trial)
    penalty = float(penalties(X, y)[mixing][step])
    model = regression.fit_rows(X, y, penalty, mixing)
    return ({"penalty": penalty, "mixing": mixing},
            mae(regression.predict_rows(model, X_validate), y_validate))
