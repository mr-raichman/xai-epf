"""L'espai de la cerca de la xarxa elàstica: la graella d'Uniejewski et al. (2016).

Les tres barreges que l'article fixa a mà, i per a cada una els 34 càstigs de la
seqüència que fa `lasso.m`, l'eina amb què l'article les ajusta: geomètrica, del càstig
més petit que deixa a zero tots els coeficients fins a `RATIO` d'ell, el valor per
defecte de l'eina, que l'article no canvia. Un sol càstig serveix les 24 hores, com a
l'article, de manera que el més gran de cada seqüència és el més gran de les 24. La
seqüència es calcula sobre les files d'entrenament.
"""
import numpy as np

from epf.models.linear import linear

# Les barreges d'Uniejewski et al. (2016), els càstigs de cada una, i la raó entre el més
# petit i el més gran, el `LambdaRatio` per defecte de `lasso.m`.
MIXINGS = (0.25, 0.5, 0.75)
STEPS = 34
RATIO = 1e-4


def penalties(X, y):
    """Per a cada barreja, els seus `STEPS` càstigs, del que deixa a zero tots els
    coeficients de les 24 hores cap avall, sobre les files donades."""
    found = {}
    for mixing in MIXINGS:
        largest = linear.largest_penalty(X, y, mixing)
        found[mixing] = np.geomspace(largest, largest * RATIO, STEPS)
    return found


def space(trial):
    """El punt de la graella d'una prova: la barreja i el pas de la seva seqüència."""
    return (trial.suggest_categorical("mixing", MIXINGS),
            trial.suggest_categorical("step", tuple(range(STEPS))))
