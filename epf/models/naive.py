"""Prediccions ingènues de Lago et al. (2021), el terra que hi ha sota la referència.

No s'ajusta res: cada una copia una hora que ja es coneixia. Totes tres tenen la
signatura que espera el camí endavant, i per això reben `y` sense mirar-lo.
"""
from epf.data.calendar import HOUR_COLUMNS

PREVIOUS_DAY = 1
PREVIOUS_WEEK = 7

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior, perquè el cap de setmana no s'assembla als dies feiners. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)


def _lag(X, day, lag):
    return X.loc[day, [f"lag{lag}_{hour}" for hour in HOUR_COLUMNS]].to_numpy(dtype=float)


def previous_day(X, y, day):
    """La mateixa hora del dia anterior."""
    return _lag(X, day, PREVIOUS_DAY)


def previous_week(X, y, day):
    """La mateixa hora de fa una setmana."""
    return _lag(X, day, PREVIOUS_WEEK)


def similar_day(X, y, day):
    """El dia anterior de dimarts a divendres, i la setmana anterior la resta."""
    lag = PREVIOUS_DAY if day.dayofweek in WEEKDAYS_FROM_YESTERDAY else PREVIOUS_WEEK
    return _lag(X, day, lag)
