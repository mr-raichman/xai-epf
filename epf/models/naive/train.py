"""Les tres prediccions ingènues de Lago et al. (2021), i la seva passada.

No s'ajusta res: cada una copia un dia que ja es coneixia. Es puntuen i no s'expliquen.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.metrics import scored, scored_distribution
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)

PREVIOUS_WEEK = 7
PREVIOUS_DAY = 1

# Les parts de la seed d'on la predicció ingènua pren els errors per fer-se
# probabilística: el que Marcjasz et al. (2023) diuen dins la mostra, l'entrenament i la
# validació, on la còpia no tria res.
IN_SAMPLE = ("training", "validation")


def _copied(price, days, lag):
    return price.shift(lag, freq="D").reindex(days).to_numpy(dtype=float)


def similar_day(price, days):
    """Torna el dia anterior de dimarts a divendres, i el de fa una setmana la resta."""
    yesterday = days.dayofweek.isin(WEEKDAYS_FROM_YESTERDAY)
    return np.where(
        yesterday[:, None], previous_day(price, days), previous_week(price, days)
    )


def previous_week(price, days):
    """Torna el preu de la mateixa hora de fa una setmana, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_WEEK)


def previous_day(price, days):
    """Torna el preu de la mateixa hora del dia anterior, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_DAY)


def bootstrapped(copied, errors, probabilities):
    """Torna els quantils de la còpia amb els seus errors dins la mostra.

    `copied` és `(dies, hores)`, `errors` `(dies de la mostra, hores)` i el que torna
    `(dies, hores, probabilitats)`. Marcjasz et al. (2023) sumen a la còpia els errors
    d'un dia sencer, trets a l'atzar amb reemplaçament; aquí es prenen a cada hora els
    quantils exactes dels seus errors, la inversa de la distribució empírica, on van a
    parar els de les mostres com més n'hi ha.
    """
    spread = np.quantile(errors, probabilities, axis=0, method="inverted_cdf")
    return copied[:, :, None] + spread.T


# Cada predicció és una família, amb la funció que copia i els dies que copia, que és el
# que en diu el registre.
FORECASTS = {
    "naive_similar_day": (
        similar_day,
        {
            "lag": PREVIOUS_DAY,
            "weekdays_from_yesterday": WEEKDAYS_FROM_YESTERDAY,
            "lag_otherwise": PREVIOUS_WEEK,
        },
    ),
    "naive_previous_week": (previous_week, {"lag": PREVIOUS_WEEK}),
    "naive_previous_day": (previous_day, {"lag": PREVIOUS_DAY}),
}

# La que Marcjasz et al. (2023) fan probabilística, amb els errors de la mostra.
BOOTSTRAPPED = "naive_similar_day"

# Cada família, com la planifica una passada: els dies que copia, i cap cerca.
FAMILIES = {
    name: {
        "fixed": numbers
        | ({"bootstrap_from": IN_SAMPLE} if name == BOOTSTRAPPED else {}),
        "kept": 0,
        "searched_as": name,
    }
    for name, (_, numbers) in FORECASTS.items()
}


def _fitted(family, price, y, parts):
    """Copia la predicció de la família als dies de prova de la matriu.

    La del dia similar torna també la seva distribució puntuada, i els errors dels dies
    de `IN_SAMPLE`, que es desen.
    """
    forecast, _ = FORECASTS[family]
    # Els dies de prova de la matriu, sobre els quals es puntuen totes les famílies.
    days = y.index[y.index.isin(parts.test)]
    copied = forecast(price, days)
    actual = y.loc[days]
    if family != BOOTSTRAPPED:
        return Fitted(copied.ravel(), actual.to_numpy().ravel(), {})
    sample = y.index[
        np.any([y.index.isin(getattr(parts, part)) for part in IN_SAMPLE], axis=0)
    ]
    errors = y.loc[sample].to_numpy() - forecast(price, sample)
    scores = scored_distribution(
        actual, lambda probabilities: bootstrapped(copied, errors, probabilities)
    )
    return Fitted(
        copied.ravel(), actual.to_numpy().ravel(), {"errors": errors}, scores=scores
    )


def train(family, folder, configurations, explained_days):
    """Copia, puntua i desa cada configuració del pla que encara no hi és.

    `configurations` és el pla de la família per seed, i `explained_days` no hi fa res,
    perquè la família no explica res.
    """
    series = matrix.series()
    # El preu es copia de la sèrie, i no de la matriu, on la setmana anterior no és cap
    # entrada.
    price, y = series[matrix.PRICE], matrix.matrices(series).actual
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            fitted = _fitted(family, price, y, parts)
            scores = scored(fitted.predicted, fitted.actual) | (fitted.scores or {})
            save_model(path, parts, None, fitted, scores, rank)
