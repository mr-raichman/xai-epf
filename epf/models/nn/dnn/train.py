"""La passada de la xarxa de punt de Marcjasz et al. (2023), de la matriu diària als
seus fitxers: la distribucional amb la sortida canviada, ajustada sobre les mateixes
files i aturada igual. Es puntua i no s'explica.
"""

import time

import keras

from epf import paths
from epf.contract import matrix
from epf.contract.metrics import scored
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.models.nn import network
from epf.models.nn.hyperparameters import FIXED, copied
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# La sortida de la xarxa de punt de l'article: els 24 preus tal com surten, i l'error
# absolut, la pèrdua `'mae'` del seu codi. No és cap distribució, però la xarxa la
# construeix i l'ajusta igual que les altres dues, i no té quantils.
POINT = network.Distribution(
    names=("price",),
    loss=keras.losses.mean_absolute_error,
    flat=lambda raw: raw,
    point=lambda parameters: parameters[..., 0],
    quantiles=None,
)

# Com es diu la família de la xarxa de punt.
POINT_FAMILY = "dnn"

# La família, com la planifica una passada: els números de les xarxes que no es cerquen,
# els copiats de l'article mentre no hi ha cerca, i la millor configuració de cada
# llavor de la seva pròpia cerca.
FAMILIES = {
    POINT_FAMILY: {
        "fixed": FIXED,
        "placeholder": copied(POINT_FAMILY),
        "kept": 1,
        "searched_as": POINT_FAMILY,
    }
}


def _fitted(X, y, parts, numbers):
    """La xarxa de punt d'una configuració i el seu preu a les setmanes de prova."""
    training, validation, test = (X.index.isin(days) for days in parts)
    fitted = network.fit_rows(
        POINT, (X[training], y[training]), (X[validation], y[validation]), numbers
    )
    return Fitted(
        network.predict_rows(POINT, fitted, X[test]).ravel(),
        y[test].to_numpy().ravel(),
        {},
    )


def train(family, folder, configurations, explained_days):
    """La passada de la família: cada llavor ajustada, puntuada i desada. Torna les
    parelles `[llavor, configuració]` que ha desat ara."""
    X, y = matrix.matrices(matrix.series()).daily
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
            start = time.time()
            fitted = _fitted(X, y, parts, configuration["numbers"])
            scores = scored(fitted.predicted, fitted.actual)
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
            print(
                f"{family} llavor {seed} configuració {rank}: R² {scores['r2']:.3f}, "
                f"MAE {scores['mae']:.2f} EUR/MWh, {time.time() - start:.1f} s",
                flush=True,
            )
    return done
