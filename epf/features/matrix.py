"""Matriu de característiques: una fila per dia de lliurament i 24 sortides.

Tota columna d'entrada ha de ser calculable abans del tancament de la subhasta
del dia anterior al lliurament.
"""
import pandas as pd

from epf.data.calendar import HOUR_COLUMNS

# Retards del model LEAR de Lago et al. (2021): fa 1, 2, 3 i 7 dies.
PRICE_LAGS = (1, 2, 3, 7)

DAY_OF_WEEK = "day_of_week"


def calibration(X, y, day, window=None):
    """Files amb què es calibra un model per predir `day`: les anteriors, les últimes
    `window`, o totes si és None.

    Exclusiu per construcció, i és l'única implementació del tall: el dia que es
    prediu no hi entra mai, ni cap de posterior.
    """
    past = X.index < day
    features, target = X[past], y[past]
    if features.empty:
        raise ValueError(f"cap dia anterior a {day} amb què calibrar")
    if window is None:
        return features, target
    return features.iloc[-window:], target.iloc[-window:]


def build(prices, lags=PRICE_LAGS):
    """Torna (X, y) alineats per data de lliurament, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les.
    """
    if list(prices.columns) != HOUR_COLUMNS:
        raise ValueError(f"s'esperaven les columnes {HOUR_COLUMNS}, i no {list(prices.columns)}")

    blocks = [prices.shift(lag, freq="D").reindex(prices.index).add_prefix(f"lag{lag}_")
              for lag in lags]
    blocks.append(prices.index.dayofweek.to_series(index=prices.index, name=DAY_OF_WEEK))
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], prices[complete]

