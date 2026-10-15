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

from epf.parameters import (DAYS_PER_WEEK, FORECAST_LAG, HOUR_COLUMNS, HOURS_PER_DAY,
                            PREVIOUS_DAY, SCHEDULE_LAG)

HOUR = "hour"
PRICE = "price"

# La columna del conjunt de la prova de datació: els dies des del primer del període.
DAY_COUNTER = "day_counter"

# El dia de la setmana, en set indicadors i no en un enter. Els escriu la matriu i no
# cap família, perquè és l'única manera que totes vegin el mateix i que una diferència
# entre famílies sigui només d'algorisme. A una xarxa, l'enter li diu que diumenge és
# sis vegades dimarts.
DAY_OF_WEEK_COLUMNS = [f"day_of_week_{number}" for number in range(DAYS_PER_WEEK)]

# Les dues matrius d'un conjunt d'informació i el preu real amb què es comparen.
# `daily` i `hourly` són les parelles `(X, y)` que una família pren; `actual` és el
# `y` de la diària, anomenat un sol cop aquí perquè tres entrades el volen i cap no
# ha de saber que és el segon element del primer parell.
Matrices = collections.namedtuple("Matrices", "daily hourly actual")


def column(name, lag):
    """El nom d'una sèrie d'entrada al seu retard: tal qual al dia que es lliura, i amb el
    retard al darrere a qualsevol altre."""
    return f"{name}_lag{lag}" if lag else name


def build(price, lags, inputs=None):
    """Torna (X, y) alineats per data de lliurament, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les.
    `inputs` són sèries per nom, cada una `(taula, retard)` amb les 24 hores de cada dia:
    una previsió D+1 entra al retard 0, el dia que es lliura, sense desplaçar, perquè es
    publica el dia abans.
    """
    if list(price.columns) != HOUR_COLUMNS:
        raise ValueError(f"s'esperaven les columnes {HOUR_COLUMNS}, i no {list(price.columns)}")

    blocks = [price.shift(lag, freq="D").reindex(price.index).add_prefix(f"lag{lag}_")
              for lag in lags]
    blocks += [frame.shift(lag, freq="D").reindex(price.index)
               .add_prefix(f"{column(name, lag)}_")
               for name, (frame, lag) in (inputs or {}).items()]
    blocks.append(pd.DataFrame(np.eye(DAYS_PER_WEEK)[price.index.dayofweek],
                               index=price.index, columns=DAY_OF_WEEK_COLUMNS))
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], price[complete]


def per_hour(X, y, at_the_hour=()):
    """Les mateixes files repartides en una per dia i hora, amb l'hora com a columna.

    El gradient boosting comparteix un sol model per a totes les hores i és la taula
    que li cal. L'índex continua sent el dia de lliurament, repetit 24 vegades,
    perquè la partició hi segueixi triant per dia.

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


# Els noms dels conjunts, que són carpetes de les passades i de les cerques. `schedule` es
# diu com el conjunt de dades del programa, que és el que hi afegeix, i és una altra cosa.
SCHEDULE, DATE_PROBE = "schedule", "date_probe"


def tables(frame, day_ahead, scheduled):
    """Les dues matrius del conjunt: les 24 hores del preu del dia abans, les previsions
    D+1 del dia que es lliura i el programa del dia abans, cada sèrie per nom, i el dia
    de la setmana.

    Cada família pren les previsions i el programa com la seva font pren una previsió:
    la diària, que és la de les xarxes, amb les 24 hores de cada sèrie, i la per hora,
    que és la de l'arbre i la dels models d'una hora, amb la de l'hora de la fila.
    """
    inputs = ({name: (table, FORECAST_LAG) for name, table in day_ahead.items()}
              | {name: (table, SCHEDULE_LAG) for name, table in scheduled.items()})
    daily = build(frame, PREVIOUS_DAY, inputs)
    at_the_hour = tuple(column(name, lag) for name, (_, lag) in inputs.items())
    return Matrices(daily, per_hour(*daily, at_the_hour=at_the_hour), daily[1])


def date_probe(frame, day_ahead, scheduled, start):
    """El conjunt amb un comptador de dies, els que han passat des de `start`: la prova
    de datació de Trebbien et al. (2023), que treu el que un model pot fer amb la data i
    res més. Només la taula per hora, perquè només l'arbre l'ajusta."""
    rows, target = tables(frame, day_ahead, scheduled).hourly
    counter = (rows.index - pd.Timestamp(start)).days
    return rows.assign(**{DAY_COUNTER: counter.to_numpy()}), target
