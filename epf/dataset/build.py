"""El dataset, `output/data/dataset.csv`, fet dels fitxers d'ESIOS baixats."""

import datetime
import json
import sys

import numpy as np
import pandas as pd

from epf import Refusal, paths
from epf.dataset import esios, figures
from epf.dataset.check import checks, outliers
from epf.dataset.esios import Indicator
from epf.parameters import DAYS_PER_WEEK, DEVELOPMENT, HOURS_PER_DAY

# Les dues columnes que diuen de quin dia local i de quina hora és cada fila.
DAY, HOUR = "day", "hour"

# El primer i l'últim dia que es baixen i que té el dataset: el període i, abans, la
# setmana que en llegeix el retard més llarg, el de les prediccions ingènues que copien
# la setmana anterior.
SPAN = (DEVELOPMENT[0] - datetime.timedelta(days=DAYS_PER_WEEK), DEVELOPMENT[1])

# Les hores de l'apagada, que ESIOS no publica en cap indicador del programa, l'àncora
# inclosa. És l'única llacuna que es declara, i no s'omple.
BLACKOUT = pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")

# Cada indicador, pel nom de la seva carpeta a `output/downloads/`, i com el serveix
# ESIOS. El preu, el del mercat diari en EUR/MWh, es tria entre les sis geografies que
# porta el seu fitxer. Les previsions D+1 de demanda, eòlica i fotovoltaica, en MW, són
# de Red Eléctrica, i la solar és només la fotovoltaica. El programa base de
# funcionament (PBF), en MWh per hora, es demana com la suma de cada hora: des del
# 2025-10-01 ESIOS el serveix com a energia per quart d'hora, i la mitjana en seria la
# quarta part. L'eòlica programada, que no val mai zero, és l'àncora de la resta del
# programa: una hora que l'àncora publica i un altre indicador no, hi val zero, també
# abans del primer dia de l'indicador i passat l'últim. L'hidràulica és la de les
# unitats de gestió hidràulica (UGH) i la de fora d'elles, i el bombeig, el seu
# turbinatge.
INDICATORS = {
    "price": Indicator(600, "average", "España", False, None, ()),
    "demand_fcst": Indicator(1775, "average", "Península", True, None, ()),
    "wind_fcst": Indicator(1777, "average", "Península", True, None, ()),
    "solar_fcst": Indicator(1779, "average", "Península", True, None, ()),
    "wind_sched": Indicator(12, "sum", "Península", True, None, BLACKOUT),
    "nuclear_sched": Indicator(4, "sum", "Península", True, "wind_sched", BLACKOUT),
    "anthracite_sched": Indicator(7, "sum", "Península", True, "wind_sched", BLACKOUT),
    "subbituminous_sched": Indicator(
        8, "sum", "Península", True, "wind_sched", BLACKOUT
    ),
    "combined_cycle_sched": Indicator(
        9, "sum", "Península", True, "wind_sched", BLACKOUT
    ),
    "gas_steam_sched": Indicator(11, "sum", "Península", True, "wind_sched", BLACKOUT),
    "cogeneration_sched": Indicator(
        17, "sum", "Península", True, "wind_sched", BLACKOUT
    ),
    "hydro_ugh_sched": Indicator(1, "sum", "Península", True, "wind_sched", BLACKOUT),
    "hydro_non_ugh_sched": Indicator(
        2, "sum", "Península", True, "wind_sched", BLACKOUT
    ),
    "pumped_storage_sched": Indicator(
        3, "sum", "Península", True, "wind_sched", BLACKOUT
    ),
}

# Cada columna del dataset, en el seu ordre, amb els indicadors que suma, en l'ordre en
# què se sumen. L'eòlica programada no és cap columna: només fa d'àncora.
SERIES = {
    "price": ("price",),
    "demand_fcst": ("demand_fcst",),
    "wind_fcst": ("wind_fcst",),
    "solar_fcst": ("solar_fcst",),
    "nuclear_sched": ("nuclear_sched",),
    "coal_gas_sched": (
        "anthracite_sched",
        "subbituminous_sched",
        "combined_cycle_sched",
        "gas_steam_sched",
        "cogeneration_sched",
    ),
    "hydro_sched": ("hydro_ugh_sched", "hydro_non_ugh_sched", "pumped_storage_sched"),
}


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
    seu nom.
    """
    anchors = {indicator.anchor for indicator in INDICATORS.values()} - {None}
    published, frames, read = {}, {}, []
    for name, indicator in INDICATORS.items():
        found = esios.files(directory / name, SPAN)
        try:
            values = esios.admissible(esios.values(found), indicator)
            if name in anchors:
                published[name] = esios.published_instants(values)
            anchor = published[indicator.anchor] if indicator.anchor else None
            frames[name] = esios.hourly(values, anchor)
        except Refusal as refusal:
            raise Refusal(f"l'indicador {name} té {refusal}") from None
        read += found
    return frames, read


def series(frames):
    """Torna cada sèrie de `SERIES`, la suma dels seus indicadors, en dies per hores.

    Va del seu primer dia dins de `SPAN` a l'últim sense cap dia que falti: un dia que
    cap dels seus indicadors va publicar, com el de l'apagada, hi queda buit.
    """
    found = {}
    for name, parts in SERIES.items():
        total = sum((frames[part] for part in parts[1:]), frames[parts[0]])
        total = total.loc[pd.Timestamp(SPAN[0]) : pd.Timestamp(SPAN[1])]
        found[name] = total.reindex(
            pd.date_range(total.index[0], total.index[-1], freq="D", name="day")
        )
    return found


def gaps(name):
    """Torna les hores d'una sèrie que la font no va publicar."""
    return pd.DatetimeIndex(
        sorted({hour for part in SERIES[name] for hour in INDICATORS[part].gaps})
    )


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
        {name: gaps(name) for name in SERIES},
    )
    partial.replace(paths.DATASET)
    record = {
        "files": {
            path.relative_to(paths.DOWNLOADS).as_posix(): paths.digest(path)
            for path in sorted(read)
        },
        "dataset": paths.digest(paths.DATASET),
    }
    paths.write_whole(paths.DATASET_RECORD, json.dumps(record, indent=2))


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
        title, _ = figures.LABELS[name]
        outliers.draw(name, title, frame, *outliers.report(name, frame))
