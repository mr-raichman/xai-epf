"""Les puntuacions de tots els models: la R-quadrat i el MAE, el CRPS d'una predicció
distribucional, el test de cobertura dels seus intervals, i la guarda que ha de passar
el preu real.

Res d'aquí tria cap paràmetre: escollir es fa sobre la validació, no sobre la prova.
"""

import numpy as np
from scipy import stats
from scipy.special import xlogy

# Els percentils sobre els quals s'aproxima el CRPS. Marcjasz et al. (2023) en fan
# servir 99, que és el pinball mitjà sobre tots els centils interiors.
PERCENTILES = np.arange(1, 100) / 100

# Els intervals de predicció que es comproven, per cobertura nominal en percentatge, amb
# els quantils que en són els extrems: els de Marcjasz et al. (2023), centrats com els
# defineixen Nowotarski i Weron (2018), del quantil alfa/2 a l'1 - alfa/2.
INTERVALS = {50: (0.25, 0.75), 90: (0.05, 0.95)}

# Nivell de significació del test de Kupiec, el de Marcjasz et al. (2023).
SIGNIFICANCE = 0.05


def refuse_missing(frame, what):
    """Falla si la matriu porta cap valor absent, anomenant on és el primer.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés: es refusa abans, dient on és.
    """
    missing = frame.isna()
    if not missing.to_numpy().any():
        return
    flat = missing.stack()
    row, column = next(iter(flat[flat].index))
    raise ValueError(
        f"{int(missing.to_numpy().sum())} valors absents a {what}, "
        f"el primer a {row} hora {column}"
    )


def _checked(actual, quantiles):
    """El preu real, `(dies, hores, 1)`, si ni ell ni els quantils predits porten cap
    valor que no sigui un número."""
    refuse_missing(actual, "el preu real")
    if not np.isfinite(quantiles).all():
        raise ValueError("els quantils predits porten valors no finits")
    return actual.to_numpy()[:, :, None]


def crps(actual, quantiles, probabilities=PERCENTILES):
    """CRPS d'una predicció distribucional, pel pinball mitjà sobre els percentils.

    El CRPS és la mesura pròpia d'una predicció distribucional: diu si tota la
    distribució és bona i no només el seu centre, de manera que un model que encerta el
    preu i s'equivoca en la incertesa en paga el preu. S'aproxima com fa Marcjasz et
    al. (2023), amb el pinball mitjà sobre 99 percentils, que és l'aproximació
    habitual.

    El pinball d'un quantil `q` és `(1-q)(Q - y)` quan el preu real queda per sota del
    quantil predit i `q(y - Q)` quan queda per sobre: penalitza més quedar-se curt als
    quantils alts i llarg als baixos, que és el que fa que una distribució massa
    estreta surti pitjor que una massa ampla amb el mateix centre.

    `quantiles` és `(dies, hores, percentils)` i `actual` la matriu de preus dels
    mateixos dies, en el mateix ordre.
    """
    difference = _checked(actual, quantiles) - quantiles
    return float(
        np.maximum(probabilities * difference, (probabilities - 1) * difference).mean()
    )


def kupiec(actual, bounds, coverage):
    """El valor p del test de cobertura incondicional de Kupiec (1995) a cada hora.

    `bounds` són els extrems de l'interval, `(dies, hores, 2)`, i `coverage` la seva
    cobertura nominal. Un preu dins l'interval, extrems inclosos, n'és un encert, i el
    test compara la fracció d'encerts amb la nominal per la raó de versemblances
    `-2 log((1 - c)^n0 c^n1 / ((1 - p)^n0 p^n1))`, una khi quadrat d'un grau, amb `n0`
    les fallades, `n1` els encerts i `p` la fracció d'encerts (Nowotarski i Weron,
    2018). Es fa hora a hora perquè les 24 d'un dia surten de la mateixa informació.

    `xlogy` dóna zero al terme d'un recompte zero, que n'és el límit: amb `log`, una
    hora que no falla mai donaria un NaN, i un NaN no passa mai el test sense dir per
    què.
    """
    price = _checked(actual, bounds)[..., 0]
    hits = (bounds[..., 0] <= price) & (price <= bounds[..., 1])
    inside = hits.sum(axis=0)
    outside = len(hits) - inside
    observed = inside / len(hits)
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
    """La R-quadrat i el MAE sobre totes les files de prova juntes."""
    error = np.asarray(predicted) - np.asarray(actual)
    spread = np.asarray(actual) - np.mean(actual)
    return {
        "r2": float(1 - np.sum(error**2) / np.sum(spread**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def scored_distribution(actual, quantiles):
    """El CRPS sobre totes les hores de prova i, de cada interval, quantes hores passen
    el test de Kupiec, que és el que en reporten Marcjasz et al. (2023).

    `actual` és la matriu del preu real de les files de prova i `quantiles` la funció
    que en dóna els quantils predits a les probabilitats que se li demanen, `(dies,
    hores, probabilitats)`.
    """
    found = {"crps": crps(actual, quantiles(PERCENTILES))}
    for level, bounds in INTERVALS.items():
        values = kupiec(actual, quantiles(np.array(bounds)), level / 100)
        found[f"kupiec_{level}"] = int((values >= SIGNIFICANCE).sum())
    return found
