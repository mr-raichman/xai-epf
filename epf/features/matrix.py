"""Matriu de característiques: una fila per dia de lliurament i 24 sortides.

Tota columna d'entrada ha de ser calculable abans del tancament de la subhasta
del dia anterior al lliurament.
"""
import pandas as pd

from epf.config.market import HOURS_PER_DAY
from epf.data.omie import HOUR_COLUMNS

# Retards del model LEAR de Lago et al. (2021): fa 1, 2, 3 i 7 dies.
PRICE_LAGS = (1, 2, 3, 7)

DAY_OF_WEEK = "day_of_week"


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


def feature_names(lags=PRICE_LAGS):
    """Noms de les columnes de `build`, en el mateix ordre."""
    names = [f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS]
    return names + [DAY_OF_WEEK]


def horizon_names():
    """Noms de les 24 sortides, una per hora del dia de lliurament."""
    return list(HOUR_COLUMNS[:HOURS_PER_DAY])
