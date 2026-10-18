"""Les dues matrius que llegeixen els models, alineades per dia de lliurament.

Cap columna d'entrada no porta el preu que explica ni res del que passa el dia del
lliurament: porta el que es podia esperar a la subhasta del dia anterior, a les 12:00,
encara que es publiqui més tard.
"""

import collections
import hashlib
import json

import numpy as np
import pandas as pd

from epf import Refusal, paths
from epf.parameters import (
    DAYS_PER_WEEK,
    FORECAST_LAG,
    HOUR_COLUMNS,
    HOURS_PER_DAY,
    PREVIOUS_DAY,
    SCHEDULE_LAG,
)

DAY = "day"
HOUR = "hour"
PRICE = "price"

# El conjunt d'informació: cada sèrie del conjunt de dades que entra a les matrius, en
# l'ordre de les seves columnes, amb els dies abans del lliurament en què hi entra. El
# preu, que és també el que es prediu, hi entra amb les 24 hores de cada retard.
INFORMATION = {
    PRICE: PREVIOUS_DAY,
    "demand_fcst": (FORECAST_LAG,),
    "wind_fcst": (FORECAST_LAG,),
    "solar_fcst": (FORECAST_LAG,),
    "nuclear_sched": (SCHEDULE_LAG,),
    "coal_gas_sched": (SCHEDULE_LAG,),
    "hydro_sched": (SCHEDULE_LAG,),
}

# El dia de la setmana, en set indicadors, com els llegeixen l'arbre i la xarxa
# elàstica; les xarxes en fan un sol enter, a `models/nn/network.py`.
DAY_OF_WEEK_COLUMNS = [f"day_of_week_{number}" for number in range(DAYS_PER_WEEK)]

# Les dues matrius d'un conjunt d'informació i el preu real amb què es comparen.
# `daily` i `hourly` són les parelles `(X, y)` que una família pren; `actual` és el
# `y` de la diària.
Matrices = collections.namedtuple("Matrices", "daily hourly actual")


def column(name, lag):
    """Torna el nom d'una sèrie a un retard, `price_lag1`, o tal qual al retard 0."""
    return f"{name}_lag{lag}" if lag else name


def build(series, information):
    """Torna la matriu diària, `(X, y)` amb el preu com a `y`, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les. `series`
    són les sèries per nom, cada una amb les 24 hores de cada dia, i `information` diu
    quines entren i a quins retards, com `INFORMATION`.
    """
    price = series[PRICE]
    if list(price.columns) != HOUR_COLUMNS:
        raise ValueError(
            f"s'esperaven les columnes de {HOUR_COLUMNS[0]} a {HOUR_COLUMNS[-1]}, i no "
            f"{', '.join(map(str, price.columns))}"
        )

    blocks = [
        series[name]
        .shift(lag, freq="D")
        .reindex(price.index)
        .add_prefix(f"{column(name, lag)}_")
        for name, lags in information.items()
        for lag in lags
    ]
    blocks.append(
        pd.DataFrame(
            np.eye(DAYS_PER_WEEK)[price.index.dayofweek],
            index=price.index,
            columns=DAY_OF_WEEK_COLUMNS,
        )
    )
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], price[complete]


def per_hour(X, y, at_the_hour=()):
    """Reparteix cada fila en una per hora del dia, amb l'hora com a columna.

    L'índex continua sent el dia de lliurament, repetit 24 vegades, perquè la partició
    hi segueixi triant per dia.

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


def _digest(path):
    """Torna la SHA-256 d'un fitxer, en hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_stale():
    """Refusa un conjunt de dades que no hi és o que no és el dels fitxers d'ara.

    Els fitxers que va llegir han de ser com eren, i `dataset.csv` com es va escriure.
    Una baixada torna a escriure sempre el fitxer de l'any en curs, de manera que una
    baixada que no s'ha seguit d'`epf dataset` també es refusa.
    """
    again = 'recrear amb "uv run epf dataset"'
    if not (paths.DATASET.is_file() and paths.DATASET_RECORD.is_file()):
        raise Refusal(
            f"no hi ha cap conjunt de dades a {paths.relative(paths.DATA)}/; crear amb "
            f'"uv run epf dataset"'
        )
    record = json.loads(paths.DATASET_RECORD.read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in record["files"].items()
        if not (paths.DOWNLOADS / name).is_file()
        or _digest(paths.DOWNLOADS / name) != digest
    ]
    if changed:
        first = paths.relative(paths.DOWNLOADS / changed[0])
        counted = (
            f"1 dels fitxers baixats que va llegir ha canviat o no hi és, {first}"
            if len(changed) == 1
            else f"{len(changed)} dels fitxers baixats que va llegir han canviat o no "
            f"hi són, el primer {first}"
        )
        raise Refusal(f"el conjunt de dades és desfasat: {counted}; {again}")
    if _digest(paths.DATASET) != record["dataset"]:
        raise Refusal(
            f"{paths.relative(paths.DATASET)} ha canviat des que es va fer; {again}"
        )


def series():
    """Llegeix cada sèrie del conjunt de dades, pel seu nom, en dies per hores.

    Cada una va del primer al darrer dia amb algun valor: un dia buit fora del tram
    d'una sèrie faria una fila amb el preu que s'ha de predir absent. Refusa un conjunt
    de dades que no hi és o que és desfasat.
    """
    _refuse_stale()
    # `round_trip`, l'únic lector de pandas que torna exactament el que es va escriure:
    # el de per defecte canvia `0.1 + 0.2`.
    table = pd.read_csv(paths.DATASET, float_precision="round_trip", encoding="utf-8")
    wide = table.pivot(index=DAY, columns=HOUR)
    days = pd.DatetimeIndex(pd.to_datetime(wide.index, format="%Y-%m-%d"), name=DAY)
    found = {}
    for name in table.columns.drop([DAY, HOUR]):
        frame = pd.DataFrame(wide[name].to_numpy(), index=days, columns=HOUR_COLUMNS)
        present = frame.index[frame.notna().any(axis="columns")]
        found[name] = frame.loc[present[0] : present[-1]]
    return found


def matrices(series):
    """Torna les dues matrius del conjunt d'informació, `INFORMATION`, i el preu real.

    Cada família pren les previsions i el programa com la seva font pren una previsió:
    la diària, la de les xarxes, amb les 24 hores de cada sèrie, i la per hora, la de
    l'arbre i la dels models d'una hora, amb la de l'hora de la fila. El preu del dia
    abans hi és amb les seves 24 hores a totes dues.
    """
    daily = build(series, INFORMATION)
    at_the_hour = tuple(
        column(name, lag)
        for name, lags in INFORMATION.items()
        if name != PRICE
        for lag in lags
    )
    return Matrices(daily, per_hour(*daily, at_the_hour=at_the_hour), daily[1])


def write(found, folder):
    """Desa a `folder` les dues matrius, `found`, amb el preu que se'n prediu.

    A la diària, el del dia, de `price_h00` a `price_h23`, i a la per hora, el de
    l'hora, `price`. Una que ja hi és no es torna a escriure: una passada que es
    continua és del mateix codi i del mateix conjunt de dades que la van començar.
    """
    (X, y), (rows, target) = found.daily, found.hourly
    for name, frame in (
        ("daily", X.join(y.add_prefix(f"{PRICE}_"))),
        ("hourly", rows.assign(**{PRICE: target})),
    ):
        path = folder / paths.MATRIX.format(name=name)
        if not path.exists():
            paths.write_whole(path, frame.to_csv(index_label=DAY, lineterminator="\n"))
