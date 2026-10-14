"""Les tres prediccions ingènues de Lago et al. (2021), el terra que hi ha sota la referència.

No s'ajusta res: cada una copia les 24 hores d'un dia que ja es coneixia. Es copien de la
sèrie del preu i no de la matriu, perquè la setmana anterior no és cap entrada.
"""
import numpy as np

PREVIOUS_DAY = 1
PREVIOUS_WEEK = 7

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior, perquè el cap de setmana no s'assembla als dies feiners. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)


def _copied(price, days, lag):
    return price.shift(lag, freq="D").reindex(days).to_numpy(dtype=float)


def previous_day(price, days):
    """La mateixa hora del dia anterior, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_DAY)


def previous_week(price, days):
    """La mateixa hora de fa una setmana, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_WEEK)


def similar_day(price, days):
    """El dia anterior de dimarts a divendres, i la setmana anterior la resta."""
    yesterday = days.dayofweek.isin(WEEKDAYS_FROM_YESTERDAY)
    return np.where(yesterday[:, None], previous_day(price, days),
                    previous_week(price, days))


# Cada predicció és una família de la passada, amb el nom de la seva carpeta, i els dies
# que copia, que és el que en diu el registre.
FORECASTS = {
    "naive_previous_day": (previous_day, {"lag": PREVIOUS_DAY}),
    "naive_previous_week": (previous_week, {"lag": PREVIOUS_WEEK}),
    "naive_similar_day": (similar_day, {"lag": PREVIOUS_DAY,
                                        "weekdays_from_yesterday": WEEKDAYS_FROM_YESTERDAY,
                                        "lag_otherwise": PREVIOUS_WEEK}),
}
