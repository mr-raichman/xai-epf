"""Matrius de característiques, totes dues alineades per dia de lliurament.

La diària té una fila per dia i 24 sortides; la per hora té una fila per dia i
hora, amb l'hora com a columna.

Tota columna d'entrada ha de ser calculable abans del tancament de la subhasta
del dia anterior al lliurament.
"""
import numpy as np
import pandas as pd

from epf.config.market import HOURS_PER_DAY
from epf.data.calendar import HOUR_COLUMNS

DAY_OF_WEEK = "day_of_week"
HOUR = "hour"
PRICE = "price"


def calibration(X, y, day, window=None):
    """Files amb què es calibra un model per predir `day`: les anteriors, les dels
    últims `window` dies, o totes si és None.

    La finestra compta dies de calibratge: a la matriu diària és una fila i a la
    taula per hora en són 24, de manera que el mateix número posa totes les
    famílies sobre els mateixos dies.

    Exclusiu per construcció, i és l'única implementació del tall: el dia que es
    prediu no hi entra mai, ni cap de posterior.
    """
    past = X.index < day
    features, target = X[past], y[past]
    if features.empty:
        raise ValueError(f"cap dia anterior a {day} amb què calibrar")
    if window is None:
        return features, target
    first = features.index.unique()[-window:][0]
    kept = features.index >= first
    return features[kept], target[kept]


def build(prices, lags):
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


def per_hour(X, y):
    """Les mateixes files repartides en una per dia i hora, amb l'hora com a columna.

    El gradient boosting comparteix un sol model per a totes les hores i és la taula
    que li cal. L'índex continua sent el dia de lliurament, repetit 24 vegades,
    perquè `calibration` i el camí endavant hi segueixin triant per dia.
    """
    rows = X.loc[X.index.repeat(HOURS_PER_DAY)]
    rows[HOUR] = np.tile(np.arange(HOURS_PER_DAY), len(X))
    return rows, pd.Series(y.to_numpy().reshape(-1), index=rows.index, name=PRICE)
