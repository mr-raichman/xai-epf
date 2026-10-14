"""El conjunt del programa: la generació programada del dia (PBF) per hores, en tres grups.

El programa base de funcionament de cada tecnologia es tanca el dia abans i tothom el té
abans del tancament de la subhasta de l'endemà. Cada indicador d'ESIOS té la seva carpeta
dins de `paths.SCHEDULE`, perquè un fitxer d'ESIOS només porta el període al nom. Els nou
que se sumen no són a `esios.SERIES`; l'àncora sí, perquè també és la de les sèries del
programa que cap conjunt no ha fet seves, i viu en totes dues carpetes.

Des del 2025-10-01 ESIOS serveix el programa com a energia per quart d'hora, de manera
que es baixa amb el servidor sumant els quarts de cada hora: cada hora és l'energia de
l'hora a tots dos costats del canvi, i no la quarta part a partir d'aquell dia.

Executar el mòdul fa existir el conjunt: el baixa sencer i diu què queda a disc. `load`
passa el contracte de `contract.py` a cada grup abans de tornar-lo. El que només es pot
preguntar a ESIOS és a `audit.py`.
"""
import pathlib

import pandas as pd

from epf.data import esios
from epf.data.contract import refuse_incoherent
from epf.paths import SCHEDULE

# El nom del conjunt, que el contracte porta quan refusa un dels seus grups.
NAME = "schedule"

# Cada grup, pel nom amb què el coneix el codi, i els indicadors del PBF que suma: la
# nuclear; el carbó, l'antracita i el subbituminós, amb el cicle combinat, la turbina de
# vapor de gas natural i la cogeneració de gas natural; i l'hidràulica amb el turbinatge
# de bombeig.
GROUPS = {"nuclear": (4,), "coal_gas": (7, 8, 9, 11, 17), "hydro": (1, 2, 3)}

# L'àncora del programa, l'eòlica programada, que no val mai zero: una hora que publica i
# una sèrie no val zero, també després de l'últim dia de la sèrie.
ANCHOR = esios.PBF_WIND

# Les hores de l'apagada, que ESIOS no publica en cap sèrie del programa, l'àncora
# inclosa. És l'única llacuna que el contracte del conjunt admet, i no s'omple.
BLACKOUT = pd.date_range("2025-04-28 12:00", "2025-04-29 23:00", freq="h")


def _folder(indicator, directory):
    return pathlib.Path(directory) / str(indicator)


def anchor_instants(directory=SCHEDULE):
    """Els instants que va publicar l'àncora."""
    return esios.published_instants(esios.values(_folder(ANCHOR, directory)))


def read(indicator, anchor, directory=SCHEDULE):
    """Una sèrie del programa en MWh per hora, en dies per hores, i les hores publicades
    cada dia. Zero on l'àncora va publicar i la sèrie no; encara sense el contracte, que
    és de `load`."""
    stored = esios.values(_folder(indicator, directory))
    return esios.hourly(esios.single_geography(stored, indicator), anchor,
                        past_last_day=True)


def load(directory=SCHEDULE):
    """Els tres grups en MWh per hora, pel seu nom, amb un dia de lliurament per fila i
    una hora per columna.

    Cada grup és la suma dels seus indicadors hora a hora, i ha passat el contracte amb
    el seu nom i l'apagada declarada: qui carrega és qui comprova.
    """
    anchor = anchor_instants(directory)
    found = {}
    for name, indicators in GROUPS.items():
        frames = [read(indicator, anchor, directory)[0] for indicator in indicators]
        group = sum(frames[1:], frames[0])
        days = pd.date_range(group.index[0], group.index[-1], freq="D", name="day")
        found[name] = refuse_incoherent(f"{NAME} {name}", group.reindex(days), BLACKOUT)
    return found


def download(directory=SCHEDULE):
    """Baixa l'àncora i les nou sèries, cadascuna a la seva carpeta, amb la suma de cada
    hora."""
    session = esios.session_for()
    for indicator in (ANCHOR, *(one for group in GROUPS.values() for one in group)):
        esios.download(indicator, _folder(indicator, directory), session, aggregation="sum")


def main():
    download()
    for name, frame in load().items():
        print(f"{name}: {len(frame)} dies, de {frame.index[0].date()} "
              f"a {frame.index[-1].date()}")
