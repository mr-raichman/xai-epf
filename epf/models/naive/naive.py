"""Prediccions ingènues de Lago et al. (2021), el terra que hi ha sota la referència.

No s'ajusta res: cada una copia una hora que ja es coneixia. Totes tres tenen la
forma que espera el camí endavant, `fit` i després `predict`, i per això `fit` torna
`None` i les prediccions reben un model que no miren. Així el camí endavant no porta
cap branca que digui quines famílies ajusten alguna cosa.
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


def weekly_frame(price):
    """La predicció ingènua setmanal de cada dia d'una sèrie de preus sencera.

    La mateixa que `previous_week`, però sobre una taula de preus i no sobre una
    matriu de característiques, perquè és el denominador del rMAE de Lago et al.
    (2021) i aquest s'ha de poder calcular d'una execució ja desada, sense anar a
    buscar l'execució de la família ingènua. Els dies que no tenen el seu dia de fa
    una setmana no hi són.
    """
    earlier = price.shift(PREVIOUS_WEEK, freq="D").reindex(price.index)
    return earlier[earlier.notna().all(axis="columns")]
