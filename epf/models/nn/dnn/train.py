"""La passada de la xarxa de punt de Marcjasz et al. (2023).

Es puntua i no s'explica.
"""

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
# construeix i l'ajusta igual que les altres dues.
POINT = network.Distribution(
    names=("price",),
    loss=keras.losses.mean_absolute_error,
    flat=lambda raw: raw,
    point=lambda parameters: parameters[..., 0],
    quantiles=None,
)

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
    """Ajusta la xarxa d'una configuració i en prediu les setmanes de prova."""
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
    """Ajusta, puntua i desa cada configuració del pla que encara no hi és.

    `configurations` és el pla de la família per llavor, i `explained_days` els dies de
    prova que s'expliquen de cada llavor, tots amb `None`. Torna les parelles `[llavor,
    configuració]` que ha desat ara.
    """
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
            fitted = _fitted(X, y, parts, configuration["numbers"])
            scores = scored(fitted.predicted, fitted.actual)
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
    return done
