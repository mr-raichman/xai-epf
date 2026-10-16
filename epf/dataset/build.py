"""El conjunt de dades, `data/dataset.csv`: les sèries que llegeixen els models,
construïdes dels fitxers d'ESIOS de `downloads/`, una fila per dia local i hora i una
columna per sèrie.

Es llegeix de dalt a baix. Les dues taules diuen què hi ha: `INDICATORS`, cada indicador
d'ESIOS que es baixa i com el serveix, i `SERIES`, cada columna del conjunt amb els
indicadors que suma. `epf download` baixa el que encara falta de cada indicador. `epf
dataset` els llegeix tots, en fa les sèries, les escriu a `data/dataset.part`, hi passa
les comprovacions de `check/checks.py` i, només si passen totes, li dona el seu nom i
escriu `data/dataset.json`, l'empremta de cada fitxer llegit i la del conjunt; després
dibuixa les figures de les dades i imprimeix l'informe de veïnatge de cada sèrie.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from epf import paths
from epf.dataset import esios, figures
from epf.dataset.check import checks, outliers
from epf.dataset.esios import Indicator
from epf.parameters import HOURS_PER_DAY

# Les dues columnes que diuen de quin dia local i de quina hora és cada fila.
DAY, HOUR = "day", "hour"

# Les hores de l'apagada, que ESIOS no publica en cap indicador del programa, l'àncora
# inclosa. És l'única llacuna que es declara, i no s'omple.
BLACKOUT = pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")

# Cada indicador, pel nom de la seva carpeta a `downloads/`, i com el serveix ESIOS. El
# preu, el del mercat diari en EUR/MWh, es tria entre els sis països que porta el seu
# fitxer. Les previsions D+1 de demanda, eòlica i fotovoltaica, en MW, són de Red
# Eléctrica, i la solar és només la fotovoltaica; la d'eòlica més fotovoltaica que pren
# Lago et al. (2021) no hi és, perquè ESIOS la publica com a 10358 i és exactament la
# suma de les dues d'aquí. El programa base de funcionament (PBF), en MWh per hora, es
# demana com la suma de cada hora: des del 2025-10-01 ESIOS el serveix com a energia per
# quart d'hora, i la mitjana en seria la quarta part. L'eòlica programada, que no val
# mai zero, és l'àncora de la resta del programa: una hora que publica i un altre
# indicador no hi val zero, també passat l'últim dia de l'indicador. L'hidràulica és la
# de les unitats de gestió hidràulica (UGH) i la de fora d'elles, i el bombeig, el seu
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

# Cada columna del conjunt de dades, en el seu ordre, amb els indicadors que suma, en
# l'ordre en què se sumen: el carbó, l'antracita i el subbituminós, amb el cicle
# combinat, la turbina de vapor de gas natural i la cogeneració de gas natural, i
# l'hidràulica amb el turbinatge de bombeig. L'eòlica programada no és cap columna:
# només fa d'àncora.
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
    """`epf download`: el que encara falta de cada indicador, a la seva carpeta."""
    session = esios.session_for()
    for name, indicator in INDICATORS.items():
        esios.download(indicator, paths.DOWNLOADS / name, session)


def indicators(directory):
    """Cada indicador en dies per hores, pel seu nom, i els fitxers que s'han llegit.

    Una àncora es llegeix abans que els indicadors que hi estan ancorats, que és l'ordre
    de la taula, i els instants que publica es guarden per a ells.
    """
    anchors = {indicator.anchor for indicator in INDICATORS.values()} - {None}
    published, frames, read = {}, {}, []
    for name, indicator in INDICATORS.items():
        found = esios.files(directory / name)
        values = esios.admissible(esios.values(found), indicator)
        if name in anchors:
            published[name] = esios.published_instants(values)
        anchor = published[indicator.anchor] if indicator.anchor else None
        frames[name] = esios.hourly(values, anchor)
        read += found
    return frames, read


def series(frames):
    """Cada sèrie de `SERIES`, la suma dels seus indicadors en el seu ordre, en dies per
    hores, del seu primer dia a l'últim sense cap dia que falti: un dia que cap dels
    seus indicadors va publicar, com el de l'apagada, hi queda buit."""
    found = {}
    for name, parts in SERIES.items():
        total = sum((frames[part] for part in parts[1:]), frames[parts[0]])
        found[name] = total.reindex(
            pd.date_range(total.index[0], total.index[-1], freq="D", name="day")
        )
    return found


def gaps(name):
    """Les hores que una sèrie declara que la font no va publicar: les dels seus
    indicadors."""
    return pd.DatetimeIndex(
        sorted({hour for part in SERIES[name] for hour in INDICATORS[part].gaps})
    )


def table(found):
    """El conjunt de dades: una fila per dia local i hora, del primer dia que té alguna
    sèrie a l'últim, i una columna per sèrie, buida fora del seu tram."""
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


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(rows, read):
    """Escriu el conjunt de dades a part, hi passa les comprovacions sobre el que s'ha
    escrit, llegit com el llegeixen els models, i, si passen totes, li dona el seu nom i
    n'escriu el registre: l'empremta de cada fitxer llegit, pel seu camí sota
    `downloads/`, i la del conjunt."""
    paths.DATA.mkdir(exist_ok=True)
    partial = paths.DATASET.with_suffix(".part")
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
            path.relative_to(paths.DOWNLOADS).as_posix(): _digest(path)
            for path in sorted(read)
        },
        "dataset": _digest(paths.DATASET),
    }
    paths.write_whole(paths.DATASET_RECORD, json.dumps(record, indent=2))


def main():
    """`epf dataset`: el conjunt de dades construït, comprovat i escrit, les seves
    figures i l'informe de veïnatge de cada sèrie."""
    try:
        frames, read = indicators(paths.DOWNLOADS)
        found = series(frames)
        write(table(found), read)
    except (FileNotFoundError, ValueError) as refused:
        print(refused)
        return 1
    print(
        f"el conjunt de dades és a {paths.DATASET}, fet de {len(read)} fitxers baixats:"
    )
    for name, frame in found.items():
        print(
            f"  {name}: {len(frame)} dies, de {frame.index[0].date()} "
            f"a {frame.index[-1].date()}"
        )
    for name, frame in found.items():
        print()
        (paths.DATA_FIGURES / name).mkdir(parents=True, exist_ok=True)
        figures.draw(name, frame)
        outliers.draw(name, frame, *outliers.report(name, frame))
    print(f"\nles figures són a {paths.DATA_FIGURES}")
    return 0
