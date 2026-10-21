"""Les puntuacions de tots els models."""

import numpy as np
from scipy import stats
from scipy.special import xlogy

from epf import Refusal

# Els percentils sobre els quals s'aproxima el CRPS. Marcjasz et al. (2023) en fan
# servir 99, que és el pinball mitjà sobre tots els centils interiors.
PERCENTILES = np.arange(1, 100) / 100

# Els intervals de predicció que es comproven, per cobertura nominal en percentatge, amb
# els quantils que en són els extrems: els de Marcjasz et al. (2023), centrats com els
# defineixen Nowotarski i Weron (2018), del quantil alfa/2 a l'1 - alfa/2.
INTERVALS = {50: (0.25, 0.75), 90: (0.05, 0.95)}

# Nivell de significació del test de Kupiec, el de Marcjasz et al. (2023).
SIGNIFICANCE = 0.05

# Distància a un extrem, en EUR/MWh, dins de la qual un preu hi és a sobre, perquè
# l'interval tancat de Nowotarski i Weron (2018) ho sigui també en coma flotant. Els
# preus són múltiples de 0.00125, i els extrems del dia similar, fets de preus, també:
# dos valors diferents se separen almenys 0.00125, i l'arrodoniment, també el d'un error
# desat en `float32`, en mou menys de 4e-5.
ON_BOUND = 1e-4


def _fail_on_absent(actual):
    """Falla si el preu real té cap valor absent, i diu quants.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés.
    """
    absent = int(np.isnan(np.asarray(actual)).sum())
    # Només un error del codi hi arriba, i n'hi ha prou amb el recompte: el dataset no
    # té cap preu absent dins del seu tram, i `matrix.series` retalla cada sèrie al seu.
    if absent:
        raise ValueError(
            "el preu real té 1 valor absent"
            if absent == 1
            else f"el preu real té {absent} valors absents"
        )


def _checked(actual, quantiles):
    """Torna el preu real, `(dies, hores, 1)`, si ell i els quantils són números.

    Falla amb un preu absent, i refusa uns quantils que no són finits.
    """
    _fail_on_absent(actual)
    if not np.isfinite(quantiles).all():
        raise Refusal("els quantils predits tenen valors no finits")
    return actual.to_numpy()[:, :, None]


def crps(actual, quantiles, probabilities=PERCENTILES):
    """Torna el CRPS, el pinball mitjà sobre els percentils, com Marcjasz et al. (2023).

    `quantiles` és `(dies, hores, percentils)` i `actual` la matriu de preus dels
    mateixos dies, en el mateix ordre.
    """
    difference = _checked(actual, quantiles) - quantiles
    return float(
        np.maximum(probabilities * difference, (probabilities - 1) * difference).mean()
    )


def kupiec(actual, bounds, coverage):
    """Torna el valor p del test de cobertura incondicional de Kupiec (1995), per hora.

    `bounds` són els extrems de l'interval, `(dies, hores, 2)`, i `coverage` la seva
    cobertura nominal. Un preu dins l'interval, extrems inclosos, n'és un encert, un a
    `ON_BOUND` d'un extrem també, i el test és la raó de versemblances de Nowotarski i
    Weron (2018), una khi quadrat d'un grau.
    """
    price = _checked(actual, bounds)[..., 0]
    hits = (bounds[..., 0] - ON_BOUND <= price) & (price <= bounds[..., 1] + ON_BOUND)
    inside = hits.sum(axis=0)
    outside = len(hits) - inside
    observed = inside / len(hits)
    # `xlogy` dona zero al terme d'un recompte zero, que n'és el límit: amb `log`, una
    # hora que no falla mai donaria un NaN, que no passa mai el test.
    statistic = -2 * (
        xlogy(outside, 1 - coverage)
        + xlogy(inside, coverage)
        - xlogy(outside, 1 - observed)
        - xlogy(inside, observed)
    )
    return stats.chi2.sf(statistic, 1)


def mae(predicted, actual):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def scored(predicted, actual):
    """Torna l'R² i el MAE de totes les files juntes.

    Falla amb un preu real absent, i refusa una predicció que no és un número finit: un
    sol NaN tornaria NaN totes dues puntuacions sense dir-ho.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    _fail_on_absent(actual)
    wrong = int((~np.isfinite(predicted)).sum())
    if wrong:
        raise Refusal(
            "la predicció té 1 valor no finit"
            if wrong == 1
            else f"la predicció té {wrong} valors no finits"
        )
    error = predicted - actual
    spread = actual - np.mean(actual)
    return {
        "r2": float(1 - np.sum(error**2) / np.sum(spread**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def scored_distribution(actual, quantiles):
    """Torna el CRPS i, de cada interval, quantes hores passen el test de Kupiec.

    És el que en reporten Marcjasz et al. (2023). `actual` és la matriu del preu real i
    `quantiles` la funció que en dona els quantils predits a les probabilitats que se li
    demanen, `(dies, hores, probabilitats)`.
    """
    found = {"crps": crps(actual, quantiles(PERCENTILES))}
    for level, bounds in INTERVALS.items():
        values = kupiec(actual, quantiles(np.array(bounds)), level / 100)
        found[f"kupiec_{level}"] = int((values >= SIGNIFICANCE).sum())
    return found
