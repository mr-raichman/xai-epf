"""El dataset, `output/data/dataset.csv`, fet dels fitxers d'ESIOS baixats."""

import datetime
import sys

import numpy as np
import pandas as pd

from epf import Refusal, paths
from epf.dataset import esios, figures
from epf.dataset.check import checks, outliers
from epf.dataset.esios import Indicator
from epf.information import INFORMATION, KINDS
from epf.parameters import DAYS_PER_WEEK, DEVELOPMENT, HOURS_PER_DAY

# Les dues columnes que diuen de quin dia local i de quina hora és cada fila.
DAY, HOUR = "day", "hour"

# La carpeta de cada indicador a `output/downloads/`, la de la seva sèrie i el seu
# número.
FOLDER = "{series}_{id}"

# L'eòlica programada, que no val mai zero, és l'àncora del programa: una hora que
# l'àncora publica i un indicador del programa no, hi val zero, també abans del primer
# dia de l'indicador i passat l'últim. No és cap sèrie.
ANCHOR_ID = 12
ANCHOR = FOLDER.format(series="wind_sched", id=ANCHOR_ID)

# Les hores de l'apagada, que ESIOS no publica en cap indicador del programa, l'àncora
# inclosa. És l'única llacuna que es declara, i no s'omple.
BLACKOUT = pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")


def span(information):
    """Torna el primer i l'últim dia que es baixen i que té el dataset.

    És el període i, abans, el retard més llarg de les sèries d'`information` o, si és
    més curt, la setmana que copia la predicció ingènua de la setmana anterior.
    """
    longest = max(lag for series in information.values() for lag in series.lags)
    before = datetime.timedelta(days=max(longest, DAYS_PER_WEEK))
    return DEVELOPMENT[0] - before, DEVELOPMENT[1]


SPAN = span(INFORMATION)


def declared():
    """Torna cada indicador, pel nom de la seva carpeta, com el serveix ESIOS.

    L'àncora és el primer, perquè `indicators` l'ha de llegir abans que els que hi estan
    ancorats.
    """
    schedule = KINDS["schedule"]
    found = {
        ANCHOR: Indicator(
            ANCHOR_ID, schedule.aggregation, schedule.geography, schedule.only, None
        )
    }
    for name, row in INFORMATION.items():
        kind = KINDS[row.kind]
        for number in row.ids:
            found[FOLDER.format(series=name, id=number)] = Indicator(
                number,
                kind.aggregation,
                kind.geography,
                kind.only,
                ANCHOR if kind.anchored else None,
            )
    return found


INDICATORS = declared()


def download():
    """Baixa el que encara falta de cada indicador, la comanda `epf download`.

    Si no falta res, ho diu en una línia sense llegir la clau ni demanar res a ESIOS. Si
    no, escriu a stderr un encapçalament, quants fitxers ha baixat de cada indicador i
    cada fitxer que ESIOS ha refusat. Si n'hi ha cap, acaba amb un refús que els compta.
    """
    if not any(esios.missing(paths.DOWNLOADS / name, SPAN) for name in INDICATORS):
        print(
            f"Els {len(INDICATORS)} indicadors ja estan baixats de {SPAN[0]} a "
            f"{SPAN[1]}",
            file=sys.stderr,
        )
        return
    session = esios.session_for()
    print("Baixant les dades d'ESIOS", file=sys.stderr)
    print(file=sys.stderr)
    refused = []
    for name, indicator in INDICATORS.items():
        written, failed = esios.download(
            indicator, paths.DOWNLOADS / name, SPAN, session
        )
        for path, status in failed:
            print(
                f"error: no s'ha pogut baixar {paths.relative(path)}: HTTP {status}",
                file=sys.stderr,
            )
        files = "fitxer baixat" if written == 1 else "fitxers baixats"
        print(f"{name}: {written} {files}", file=sys.stderr)
        refused += [path for path, _ in failed]
    if refused:
        first = paths.relative(refused[0])
        raise Refusal(
            f"no s'ha pogut baixar 1 fitxer, {first}"
            if len(refused) == 1
            else f"no s'han pogut baixar {len(refused)} fitxers, el primer {first}"
        )


def indicators(directory):
    """Llegeix cada indicador de `directory` en dies per hores, pel seu nom.

    Torna també els fitxers llegits. Una àncora ha de precedir a `INDICATORS` els
    indicadors que hi estan ancorats. Un indicador que el lector refusa es refusa amb el
    seu nom, i també un sense cap valor de la seva geografia als anys llegits, sense
    interpretar per què.
    """
    anchors = {indicator.anchor for indicator in INDICATORS.values()} - {None}
    published, frames, read = {}, {}, []
    for name, indicator in INDICATORS.items():
        found = esios.files(directory / name, SPAN)
        try:
            values = esios.admissible(esios.values(found), indicator)
        except Refusal as refusal:
            raise Refusal(f"l'indicador {name} té {refusal}") from None
        # Sense cap valor, un indicador ancorat no tindria els zeros de l'àncora, i un
        # que no ho és faria fallar `series`.
        if not values:
            raise Refusal(f"l'indicador {name} no té cap valor")
        if name in anchors:
            published[name] = esios.published_instants(values)
        anchor = published[indicator.anchor] if indicator.anchor else None
        try:
            frames[name] = esios.hourly(values, anchor)
        except Refusal as refusal:
            raise Refusal(f"l'indicador {name} té {refusal}") from None
        read += found
    return frames, read


def series(frames):
    """Torna cada sèrie, la suma dels seus indicadors, en dies per hores.

    `frames` són els indicadors, pel nom de la seva carpeta. Cada sèrie va del seu
    primer dia dins de `SPAN` a l'últim sense cap dia que falti: un dia que cap dels
    seus indicadors va publicar, com el de l'apagada, hi queda buit.
    """
    found = {}
    for name, row in INFORMATION.items():
        parts = [frames[FOLDER.format(series=name, id=number)] for number in row.ids]
        total = sum(parts[1:], parts[0])
        total = total.loc[pd.Timestamp(SPAN[0]) : pd.Timestamp(SPAN[1])]
        found[name] = total.reindex(
            pd.date_range(total.index[0], total.index[-1], freq="D", name="day")
        )
    return found


def gaps(name):
    """Torna les hores d'una sèrie que la font no va publicar."""
    if KINDS[INFORMATION[name].kind].anchored:
        return BLACKOUT
    return pd.DatetimeIndex([])


def table(found):
    """Torna el dataset, una fila per dia local i hora i una columna per sèrie.

    Va del primer dia que té alguna sèrie a l'últim, i cada columna és buida fora del
    tram de la seva sèrie.
    """
    days = pd.date_range(
        min(frame.index[0] for frame in found.values()),
        max(frame.index[-1] for frame in found.values()),
        freq="D",
    )
    rows = pd.DataFrame(
        {
            DAY: days.repeat(HOURS_PER_DAY).strftime("%Y-%m-%d"),
            HOUR: np.tile(np.arange(HOURS_PER_DAY), len(days)),
        }
    )
    for name, frame in found.items():
        rows[name] = frame.reindex(days).to_numpy().ravel()
    return rows


def write(rows, read):
    """Escriu el dataset i el seu registre, si passa les comprovacions.

    Es comprova el fitxer escrit, llegit com el llegeixen els models, abans que porti el
    seu nom, de manera que un dataset refusat deixa l'anterior com era. El registre
    porta l'empremta de cada fitxer de `read`, pel seu camí sota `output/downloads/`, i
    la del dataset.
    """
    paths.DATA.mkdir(parents=True, exist_ok=True)
    partial = paths.partial(paths.DATASET)
    rows.to_csv(partial, index=False, lineterminator="\n", encoding="utf-8")
    checks.check(
        pd.read_csv(
            partial,
            index_col=[DAY, HOUR],
            float_precision="round_trip",
            encoding="utf-8",
        ),
        {name: gaps(name) for name in INFORMATION},
    )
    partial.replace(paths.DATASET)
    record = {
        "files": {
            path.relative_to(paths.DOWNLOADS).as_posix(): paths.digest(path)
            for path in sorted(read)
        },
        "dataset": paths.digest(paths.DATASET),
    }
    paths.write_record(paths.DATASET_RECORD, record)


def main():
    """Fa el dataset, la comanda `epf dataset`.

    El comprova i l'escriu, i en desa les figures i l'estudi d'outliers de cada sèrie.
    Refusa el que refusen el lector d'ESIOS o les comprovacions.
    """
    frames, read = indicators(paths.DOWNLOADS)
    found = series(frames)
    write(table(found), read)
    for name, frame in found.items():
        (paths.DATA_FIGURES / name).mkdir(parents=True, exist_ok=True)
        figures.draw(name, frame)
        title = INFORMATION[name].title
        outliers.draw(name, title, frame, *outliers.report(name, frame))
