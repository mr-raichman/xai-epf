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
from epf.contract import split
from epf.information import KINDS, PRICE, REFERENCE_LAG, SERIES, column
from epf.parameters import DAYS_PER_WEEK, DEVELOPMENT, HOUR_COLUMNS, HOURS_PER_DAY

DAY = "day"
HOUR = "hour"

# El dia de la setmana, en set dummies, com les llegeixen l'arbre i l'elastic net; la
# xarxa neuronal en fa una sola columna, `DAY_OF_WEEK`, a `models/dnn/network.py`.
DAY_OF_WEEK = "day_of_week"
DAY_OF_WEEK_COLUMNS = [f"{DAY_OF_WEEK}_{number}" for number in range(DAYS_PER_WEEK)]

# Les dues matrius d'un information set i el que en llegeixen les famílies a més de
# les entrades. `daily` i `hourly` són les parelles `(X, y)` que una família pren, el
# `y` el preu o, amb una referència, la distància de cada hora a la del seu dia;
# `actual`, el preu de les files de la diària; `price`, la sèrie del preu sencera, de
# la qual copien les còpies; `reference`, la referència de cada fila de la diària, o
# `None`; `deleted`, els dies suprimits amb el seu motiu, com els torna
# `split.deleted`; i `information`, l'information set resolt.
Matrices = collections.namedtuple(
    "Matrices", "daily hourly actual price reference deleted information"
)


def build(series, information):
    """Torna la matriu diària, `(X, y)` amb el preu com a `y`, sense files incompletes.

    Un dia absent elimina les files que el necessiten en lloc de desplaçar-les. `series`
    són les sèries per nom, cada una amb les 24 hores de cada dia, i `information` diu
    quines entren i a quins retards, els de cada sèrie pel seu nom. Falla si el preu no
    té les columnes `HOUR_COLUMNS`.
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


def _refuse_stale():
    """Refusa un dataset que no hi és o que no és el dels fitxers d'ara.

    Els fitxers que va llegir han de ser com eren, i `dataset.csv` com es va escriure.
    Una baixada torna a escriure el fitxer de cada any que encara no és complet, de
    manera que una baixada que no s'ha seguit d'`epf dataset` també es refusa.
    """
    again = 'recrear amb "uv run epf dataset"'
    if not (paths.DATASET.is_file() and paths.DATASET_RECORD.is_file()):
        raise Refusal(
            f"no hi ha cap dataset a {paths.relative(paths.DATA)}/; crear amb "
            f'"uv run epf dataset"'
        )
    record = json.loads(paths.DATASET_RECORD.read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in record["files"].items()
        if not (paths.DOWNLOADS / name).is_file()
        or paths.digest(paths.DOWNLOADS / name) != digest
    ]
    if changed:
        first = paths.relative(paths.DOWNLOADS / changed[0])
        counted = (
            f"1 dels fitxers baixats que va llegir ha canviat o no hi és, {first}"
            if len(changed) == 1
            else f"{len(changed)} dels fitxers baixats que va llegir han canviat o no "
            f"hi són, el primer {first}"
        )
        raise Refusal(f"el dataset és desfasat: {counted}; {again}")
    if paths.digest(paths.DATASET) != record["dataset"]:
        raise Refusal(
            f"{paths.relative(paths.DATASET)} ha canviat des que es va fer; {again}"
        )


def series():
    """Llegeix cada sèrie del dataset, pel seu nom, en dies per hores.

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


def lacking(series, needed, days):
    """Torna, de cada dia de `days` al qual falta alguna columna, les que li falten.

    `needed` són les parelles `(sèrie, retard)` que llegeix cada dia, i una columna es
    diu com la diu `column`. `series` són les sèries per nom, com a `build`.
    """
    found = {}
    for name, lag in needed:
        shifted = series[name].shift(lag, freq="D").reindex(days)
        for day in days[shifted.isna().any(axis="columns").to_numpy()]:
            found.setdefault(day, []).append(column(name, lag))
    return found


def references(price, weights):
    """Torna la referència de cada dia: la mitjana del preu del dia abans.

    Cada hora hi pesa per la previsió de `weights` de la mateixa hora i el mateix dia.
    Un dia al qual falta alguna hora de qualsevol de les dues no en té.
    """
    before = price.shift(REFERENCE_LAG, freq="D")
    weighing = weights.shift(REFERENCE_LAG, freq="D").reindex(before.index).to_numpy()
    values = (before.to_numpy() * weighing).sum(axis=1) / weighing.sum(axis=1)
    return pd.Series(values, index=before.index)


def matrices(series, information_set):
    """Torna les matrius d'un information set resolt, `information_set`, com `Matrices`.

    Cada família pren les previsions i el programa com la seva font pren una previsió:
    la diària, la de la xarxa neuronal, amb les 24 hores de cada sèrie, i la per hora,
    la de l'arbre i la dels models d'una hora, amb la de l'hora de la fila. El preu, si
    hi entra, hi és amb les seves 24 hores a totes dues. Una fila sense referència, en
    un information set que en té, no hi és. Els dies suprimits són els de `DEVELOPMENT`
    que no es reparteixen, com diu `split.deleted`.
    """
    lags, weights = information_set.series, information_set.reference
    X, price = build(series, lags)
    reference = None
    if weights:
        reference = references(series[PRICE], series[weights]).reindex(X.index)
        kept = reference.notna().to_numpy()
        X, price, reference = X[kept], price[kept], reference[kept]
    y = price if reference is None else price.sub(reference, axis="index")
    at_the_hour = tuple(
        column(name, lag)
        for name, read in lags.items()
        if KINDS[SERIES[name].kind].at_the_hour
        for lag in read
    )
    # El preu del mateix dia, el que es prediu, també.
    needed = [(PRICE, 0)] + [(name, lag) for name, read in lags.items() for lag in read]
    if weights:
        needed += [(PRICE, REFERENCE_LAG), (weights, REFERENCE_LAG)]
    days = pd.date_range(*DEVELOPMENT, freq="D")
    return Matrices(
        (X, y),
        per_hour(X, y, at_the_hour=at_the_hour),
        price,
        series[PRICE],
        reference,
        split.deleted(DEVELOPMENT, lacking(series, dict.fromkeys(needed), days)),
        information_set,
    )


def partition(built, seed):
    """Torna la partició d'una seed sobre unes matrius, `built`, com `split.split`.

    Sense els seus dies suprimits, i amb el gap del seu information set.
    """
    return split.split(seed, DEVELOPMENT, built.deleted, built.information.gap)


def fingerprint(built):
    """Torna l'empremta d'unes matrius, `built`, la SHA-256 de tot el que en llegeixen.

    De cada taula, les columnes, els dies i els valors en `float64`: les dues matrius,
    el preu de les seves files, la sèrie del preu i la referència; i els dies suprimits
    amb el seu motiu.
    """
    digest = hashlib.sha256()
    (X, y), (rows, target) = built.daily, built.hourly
    for part in (X, y, rows, target, built.actual, built.price, built.reference):
        if part is None:
            digest.update(b"None")
            continue
        names = part.columns if isinstance(part, pd.DataFrame) else [part.name]
        digest.update(json.dumps([str(name) for name in names]).encode())
        digest.update(part.index.asi8.tobytes())
        digest.update(part.to_numpy(dtype=np.float64).tobytes())
    digest.update(json.dumps(paths.plain(built.deleted)).encode())
    return digest.hexdigest()


def write(found, folder):
    """Desa a `folder` les dues matrius, `found`, amb el preu que se'n prediu.

    A la diària, el del dia, de `price_h00` a `price_h23`, i a la per hora, el de
    l'hora, `price`; i, amb una referència, la del dia de cada fila, `reference`, de la
    qual la família prediu la distància. Una que ja hi és no es torna a escriure: una
    passada que es continua és de les mateixes matrius que la va començar.
    """
    X, rows = found.daily[0], found.hourly[0]
    daily = X.join(found.actual.add_prefix(f"{PRICE}_"))
    hourly = rows.assign(**{PRICE: found.actual.to_numpy().ravel()})
    if found.reference is not None:
        daily = daily.assign(reference=found.reference)
        hourly = hourly.assign(
            reference=np.repeat(found.reference.to_numpy(), HOURS_PER_DAY)
        )
    for name, frame in (("daily", daily), ("hourly", hourly)):
        path = folder / paths.MATRIX.format(name=name)
        if not path.exists():
            paths.write_whole(path, frame.to_csv(index_label=DAY, lineterminator="\n"))
