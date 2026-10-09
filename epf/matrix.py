"""Matrius de característiques, totes dues alineades per dia de lliurament.

La diària té una fila per dia i 24 sortides; la per hora té una fila per dia i
hora, amb l'hora com a columna.

Tota columna d'entrada ha de ser calculable abans del tancament de la subhasta
del dia anterior al lliurament.
"""
import collections

import numpy as np
import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.market import HOURS_PER_DAY
from epf.parameters import DAYS_PER_WEEK, LAGS

HOUR = "hour"
PRICE = "price"

# El dia de la setmana, en set indicadors i no en un enter. Els escriu la matriu i no
# cap família, perquè és l'única manera que totes vegin el mateix: mentre l'enter hi
# era, LEAR l'obria en set i la resta el prenia tal qual, de manera que una diferència
# entre famílies no era només d'algorisme. A una xarxa, l'enter li diu que diumenge és
# sis vegades dimarts.
DAY_OF_WEEK_COLUMNS = [f"day_of_week_{number}" for number in range(DAYS_PER_WEEK)]

# Les dues matrius d'un conjunt d'informació i el preu real amb què es comparen.
# `daily` i `hourly` són les parelles `(X, y)` que una família pren; `actual` és el
# `y` de la diària, anomenat un sol cop aquí perquè tres entrades el volen i cap no
# ha de saber que és el segon element del primer parell.
Matrices = collections.namedtuple("Matrices", "daily hourly actual")


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


def build(price, lags):
    """Torna (X, y) alineats per data de lliurament, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les.
    """
    if list(price.columns) != HOUR_COLUMNS:
        raise ValueError(f"s'esperaven les columnes {HOUR_COLUMNS}, i no {list(price.columns)}")

    blocks = [price.shift(lag, freq="D").reindex(price.index).add_prefix(f"lag{lag}_")
              for lag in lags]
    blocks.append(pd.DataFrame(np.eye(DAYS_PER_WEEK)[price.index.dayofweek],
                               index=price.index, columns=DAY_OF_WEEK_COLUMNS))
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], price[complete]


def per_hour(X, y):
    """Les mateixes files repartides en una per dia i hora, amb l'hora com a columna.

    El gradient boosting comparteix un sol model per a totes les hores i és la taula
    que li cal. L'índex continua sent el dia de lliurament, repetit 24 vegades,
    perquè `calibration` i el camí endavant hi segueixin triant per dia.
    """
    rows = X.loc[X.index.repeat(HOURS_PER_DAY)]
    rows[HOUR] = np.tile(np.arange(HOURS_PER_DAY), len(X))
    return rows, pd.Series(y.to_numpy().reshape(-1), index=rows.index, name=PRICE)


def price_only(frame):
    """Les dues matrius del conjunt d'informació de només preu: la diària i la per
    hora, ja alineades.

    Els 96 retards de preu i els set indicadors del dia de la setmana, que és el
    model de només preu de Lago et al. (2021). Tot és calculable molt abans que
    tanqui la subhasta del dia anterior al lliurament, perquè els preus del dia d-1
    es van publicar el d-2.
    L'any més antic que es pot entrenar és el primer del preu, perquè no hi entra
    cap sèrie que comenci més tard.

    Pren el conjunt ja carregat i no el carrega, que és d'una altra capa; `price.load`
    ja n'ha comprovat el contracte.
    """
    daily = build(frame, LAGS)
    return Matrices(daily, per_hour(*daily), daily[1])
