"""Matrius de característiques, totes dues alineades per dia de lliurament.

La diària té una fila per dia i 24 sortides; la per hora té una fila per dia i
hora, amb l'hora com a columna, i hi pren a l'hora de la fila la sèrie que s'hi pren
a l'hora.

Tota columna d'entrada ha de ser calculable abans del tancament de la subhasta
del dia anterior al lliurament.
"""
import collections

import numpy as np
import pandas as pd

from epf.calendar import HOUR_COLUMNS
from epf.market import HOURS_PER_DAY
from epf.parameters import DAYS_PER_WEEK, LAGO_FORECAST_LAGS, LAGS, PREVIOUS_DAY

HOUR = "hour"
PRICE = "price"

# La columna del conjunt de la prova de datació: els dies des del primer del període.
DAY_COUNTER = "day_counter"

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
    days = features.index.unique()
    if len(days) < window:
        # Una finestra més curta del que diu el seu nom passaria sense queixar-se: el
        # conjunt de Lago et al. (2021) comença el 2019-01-08 i la seva finestra de
        # quatre anys no s'omple fins al 2022-12-27.
        raise ValueError(f"{len(days)} dies abans de {day}, i la finestra en demana {window}")
    first = days[-window:][0]
    kept = features.index >= first
    return features[kept], target[kept]


def build(price, lags, day_ahead=None, day_ahead_lags=(0,)):
    """Torna (X, y) alineats per data de lliurament, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les.
    `day_ahead` són previsions per nom, cada una amb les 24 hores del dia que es
    lliura, i entren als retards de `day_ahead_lags`: la del retard 0 és la del dia que
    es lliura, sense desplaçar, perquè es publica el dia abans.
    """
    if list(price.columns) != HOUR_COLUMNS:
        raise ValueError(f"s'esperaven les columnes {HOUR_COLUMNS}, i no {list(price.columns)}")

    blocks = [price.shift(lag, freq="D").reindex(price.index).add_prefix(f"lag{lag}_")
              for lag in lags]
    blocks += [frame.shift(lag, freq="D").reindex(price.index)
               .add_prefix(f"{name}_lag{lag}_" if lag else f"{name}_")
               for name, frame in (day_ahead or {}).items() for lag in day_ahead_lags]
    blocks.append(pd.DataFrame(np.eye(DAYS_PER_WEEK)[price.index.dayofweek],
                               index=price.index, columns=DAY_OF_WEEK_COLUMNS))
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], price[complete]


def per_hour(X, y, at_the_hour=()):
    """Les mateixes files repartides en una per dia i hora, amb l'hora com a columna.

    El gradient boosting comparteix un sol model per a totes les hores i és la taula
    que li cal. L'índex continua sent el dia de lliurament, repetit 24 vegades,
    perquè `calibration` i el camí endavant hi segueixin triant per dia.

    Una sèrie d'`at_the_hour` hi deixa les seves 24 columnes per una de sola, el valor
    de l'hora de la fila, com Trebbien et al. (2023) prenen les previsions.
    """
    rows = X.loc[X.index.repeat(HOURS_PER_DAY)]
    rows[HOUR] = np.tile(np.arange(HOURS_PER_DAY), len(X))
    for name in at_the_hour:
        columns = [f"{name}_{hour}" for hour in HOUR_COLUMNS]
        rows = rows.drop(columns=columns)
        rows[name] = X[columns].to_numpy().reshape(-1)
    return rows, pd.Series(y.to_numpy().reshape(-1), index=rows.index, name=PRICE)


def previous_day(frame):
    """Les dues matrius del conjunt de referència de la part B: les 24 hores del preu
    del dia abans i el dia de la setmana."""
    daily = build(frame, PREVIOUS_DAY)
    return Matrices(daily, per_hour(*daily), daily[1])


def forecasts(frame, day_ahead):
    """Les dues matrius del conjunt de previsions de la part B: el de referència i les
    previsions D+1 del dia que es lliura, per nom.

    Cada família les pren com la seva font: la diària, que és la de la xarxa, amb les
    24 hores de cada previsió, i la per hora, que és la de l'arbre i la dels models
    d'una hora, amb la de l'hora de la fila. Es diu com el conjunt de dades de les
    previsions i és una altra cosa, un conjunt d'informació.
    """
    daily = build(frame, PREVIOUS_DAY, day_ahead)
    return Matrices(daily, per_hour(*daily, at_the_hour=tuple(day_ahead)), daily[1])


def date_probe(frame, day_ahead, start):
    """El conjunt de previsions amb un comptador de dies, els que han passat des de
    `start`: la prova de datació de Trebbien et al. (2023), que treu el que un model pot
    fer amb la data i res més. Només la taula per hora, perquè només l'arbre l'ajusta."""
    rows, target = forecasts(frame, day_ahead).hourly
    counter = (rows.index - pd.Timestamp(start)).days
    return rows.assign(**{DAY_COUNTER: counter.to_numpy()}), target


def lago(frame, day_ahead):
    """Les dues matrius del conjunt de Lago et al. (2021): el preu de fa 1, 2, 3 i 7 dies,
    les seves dues previsions del dia que es lliura, del dia abans i de fa una setmana,
    i el dia de la setmana. 247 columnes.

    Les dues previsions són la de demanda i la d'eòlica més fotovoltaica, la parella de
    l'article per a EPEX-DE, i la segona és la suma de les dues que porta el conjunt de
    dades. La taula per hora és la fila del dia sencera més l'hora, com a l'article.
    """
    pair = {"demand": day_ahead["demand"],
            "wind_solar": day_ahead["wind"] + day_ahead["solar"]}
    daily = build(frame, LAGS, pair, LAGO_FORECAST_LAGS)
    return Matrices(daily, per_hour(*daily), daily[1])
