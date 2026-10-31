"""Les tres prediccions ingènues de Lago et al. (2021), i la seva passada.

No s'ajusta res: cada una copia un dia que ja es coneixia. Es puntuen i no s'expliquen.
"""

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.record import Fitted, save_model

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)

WEEK_LAG = 7
DAY_LAG = 1


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
    return _copied(price, days, WEEK_LAG)


def previous_day(price, days):
    """Torna el preu de la mateixa hora del dia anterior, `(dies, hores)`."""
    return _copied(price, days, DAY_LAG)


# Cada predicció és una família, amb la funció que copia, els dies que copia, que és el
# que en diu el registre, i l'rMAE que divideix: el de cada família és sobre la còpia de
# la setmana anterior, com a Lago et al. (2021), i les altres dues en donen un de
# secundari.
FORECASTS = {
    "naive_similar_day": (
        similar_day,
        {
            "lag": DAY_LAG,
            "weekdays_from_yesterday": WEEKDAYS_FROM_YESTERDAY,
            "lag_otherwise": WEEK_LAG,
        },
        "secondary",
    ),
    "naive_previous_week": (previous_week, {"lag": WEEK_LAG}, "main"),
    "naive_previous_day": (previous_day, {"lag": DAY_LAG}, "secondary"),
}

# Cada família, com la planifica una passada: els dies que copia, i cap cerca. No se'n
# fa el test de Ljung-Box dels errors, perquè no ajusten res.
FAMILIES = {
    name: {
        "fixed": numbers,
        "kept": 0,
        "searched_as": name,
        "ljung_box_tested": False,
        "interpreted": False,
        "benchmark": True,
        "denominator": denominator,
    }
    for name, (_, numbers, denominator) in FORECASTS.items()
}


def _fitted(family, price, y, parts):
    """Copia la predicció de la família als dies de prova de la matriu.

    La que divideix l'rMAE, `"main"`, desa també el MAE i el MSE de la seva còpia als
    dies d'entrenament de la matriu, `training_mae` i `training_mse`, en EUR/MWh i en
    (EUR/MWh)².
    """
    forecast, _, denominator = FORECASTS[family]
    arrays = {}
    if denominator == "main":
        # L'escala del MASE de Hyndman i Koehler (2006) i de l'RMSSE, sobre
        # l'entrenament.
        days = y.index[y.index.isin(parts.training)]
        error = y.loc[days].to_numpy() - forecast(price, days)
        arrays = {
            "training_mae": np.mean(np.abs(error)),
            "training_mse": np.mean(error**2),
        }
    # Els dies de prova de la matriu, sobre els quals es puntuen totes les famílies.
    days = y.index[y.index.isin(parts.test)]
    return Fitted(forecast(price, days).ravel(), y.loc[days].to_numpy().ravel(), arrays)


def train(family, folder, seed, configuration, explained_days, built):
    """Copia, puntua i desa una configuració del pla d'una seed.

    `explained_days` no hi fa res, perquè la família no explica res, i `built` són les
    matrius de la passada.
    """
    # El preu es copia de la sèrie, i no de la matriu, on la setmana anterior no és cap
    # entrada.
    price, y = built.price, built.actual
    parts = matrix.partition(built, seed)
    rank = configuration["rank"]
    fitted = _fitted(family, price, y, parts)
    path = folder / paths.MODEL.format(seed=seed, rank=rank)
    save_model(path, parts, None, fitted, rank)
