"""Prediccions ingènues de Lago et al. (2021), el terra que hi ha sota la referència.

No s'ajusta res: cada una copia una hora que ja es coneixia, i per això `fit` torna
`None` i les prediccions reben un model que no miren.
"""
from epf.calendar import HOUR_COLUMNS

PREVIOUS_DAY = 1
PREVIOUS_WEEK = 7

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior, perquè el cap de setmana no s'assembla als dies feiners. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)


def fit(X, y, day):
    """No hi ha res a ajustar: una predicció ingènua copia una columna que ja hi era."""
    return None


def _lag(X, day, lag):
    return X.loc[day, [f"lag{lag}_{hour}" for hour in HOUR_COLUMNS]].to_numpy(dtype=float)


def previous_day(X, day):
    """La mateixa hora del dia anterior."""
    return _lag(X, day, PREVIOUS_DAY)


def previous_week(X, day):
    """La mateixa hora de fa una setmana."""
    return _lag(X, day, PREVIOUS_WEEK)


def similar_day(X, day):
    """El dia anterior de dimarts a divendres, i la setmana anterior la resta."""
    lag = PREVIOUS_DAY if day.dayofweek in WEEKDAYS_FROM_YESTERDAY else PREVIOUS_WEEK
    return _lag(X, day, lag)


# Els tres pronòstics ingenus són una sola família i van junts en un sol directori.
FORECASTS = {"previous_day": previous_day, "previous_week": previous_week,
             "similar_day": similar_day}
NAMES = tuple(FORECASTS)


def predict(model, X, day):
    """Els tres pronòstics ingenus del dia, per nom. El model és `None` i no es mira."""
    return {name: forecast(X, day) for name, forecast in FORECASTS.items()}
