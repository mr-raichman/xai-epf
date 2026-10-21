"""Els valors fixos de tot el projecte.

Són els d'una mesura, i un assaig en canvia els de la seva variant. Una passada els desa
tots al registre de cada família.
"""

import datetime
import hashlib
import json

from epf import Refusal


### Calendari ###

DAYS_PER_WEEK = 7

# Les hores d'un dia local i el nom de la columna de cada una.
HOURS_PER_DAY = 24
HOUR_COLUMNS = [f"h{hour:02d}" for hour in range(HOURS_PER_DAY)]


### Període i partició ###

# El període que es descriu, del primer dia de les previsions D+1 a l'últim dilluns de
# les dades baixades, en setmanes de dimarts a dilluns. Totes les tries s'hi fan.
DEVELOPMENT = (datetime.date(2019, 1, 1), datetime.date(2026, 9, 28))

# La setmana que surt del període i no és de cap part: ESIOS no publica el programa del
# 2025-04-28 a les 12:00 al 2025-04-29 a les 23:00, l'apagada, i els dies que el
# necessiten, el 29 i el 30, són d'aquesta setmana.
BLACKOUT_WEEK = (datetime.date(2025, 4, 29), datetime.date(2025, 5, 5))

# Percentatges de les setmanes que van a entrenament, a validació i a prova, els de
# Trebbien et al. (2023).
SHARES = (48, 32, 20)

# Dies que surten de l'entrenament a cada costat d'una setmana retinguda. El preu del
# dia abans és una entrada, de manera que una setmana retinguda i el dia de cada costat
# es miren l'un a l'altre.
GAP = 1

# Una partició per seed.
SEEDS = tuple(range(10))


### Information set ###

# Retards de preu de l'information set: el dia abans, les seves 24 hores.
PREVIOUS_DAY = (1,)

# El dia de cada sèrie d'entrada, en dies abans del lliurament: les previsions D+1, del
# mateix dia que es lliura, i el programa del dia abans, tancat dos dies abans i a
# l'abast de tothom abans del tancament de la subhasta.
FORECAST_LAG = 0
SCHEDULE_LAG = 1


### Ajust i cerca ###

# La seed d'on surt cada tria a l'atzar del projecte, per `seeded`.
PROJECT_SEED = 0

# Trials de la cerca de cada família a cada seed, i quants dels millors se n'expliquen:
# Trebbien et al. (2023) en guarden els deu millors i no diuen quants en proven, i
# cinquanta és el nombre del projecte.
TRIALS = 50
BEST = 10


### Explicació ###

# Dies de prova que s'expliquen de cada seed, comptats des del primer. `None` vol dir
# tots, el protocol sencer.
EXPLAINED_DAYS = None

# Files explicades de cada seed sobre les quals es calculen les interaccions de l'arbre,
# triades a l'atzar amb la seed.
INTERACTION_ROWS = 1000


### Processos ###

# Quants processos fan alhora els trials i els ajustos d'una passada.
WORKERS = 4

# Els fils de cada procés, sempre els mateixos: amb uns altres un ajust dona uns altres
# números, i amb aquests una passada dona els mateixos siguin quants siguin els
# processos.
THREADS = 4


### Assaig ###

# El que un assaig posa en lloc dels valors de la mesura, i res més, en variants: cada
# diccionari que es diu `SMOKE`, o `SMOKE_` i un número, el tria l'opció del seu nom,
# `--smoke` o `--smoke1`.
SMOKE = {"SEEDS": (0,), "EXPLAINED_DAYS": 50, "TRIALS": 2}
# Els menys dies explicats amb què l'arbre encara tria les files de les interaccions.
SMOKE_1 = {
    "SEEDS": tuple(range(2)),
    "TRIALS": 1,
    "BEST": 1,
    "EXPLAINED_DAYS": INTERACTION_ROWS // HOURS_PER_DAY + 1,
}
SMOKE_2 = {"SEEDS": tuple(range(2)), "TRIALS": 10}


def seeded(label, *parts):
    """Torna la seed d'una tria a l'atzar, feta de `PROJECT_SEED`, `label` i `parts`.

    `label` diu què es tria, i `parts` són els seus números, com la seed de la
    partició. Dues llistes diferents donen seeds independents, i la mateixa llista la
    mateixa seed a cada procés, com no faria el `hash` d'un text. Llegeix
    `PROJECT_SEED` en cridar-la. La seed és de 32 bits, els que admet
    `numpy.random.seed`.
    """
    # La SHA-256 de la llista sencera, i no `SeedSequence`, que omple amb zeros una
    # llista de menys de quatre paraules: `[a, b, c]` i `[a, b, c, 0]` en donarien una.
    text = json.dumps([PROJECT_SEED, label, *parts])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def variants():
    """Torna cada variant d'assaig per l'opció que la tria: `--smoke1`, `SMOKE_1`."""
    return {
        f"--{name.lower().replace('_', '')}": name
        for name in globals()
        if name == "SMOKE" or name.startswith("SMOKE_")
    }


def chosen(smoke):
    """Torna cada valor d'aquest fitxer pel seu nom, els de la variant d'assaig `smoke`.

    `smoke` és el nom d'una variant, `SMOKE_1`, o fals per a una mesura. Refusa una
    variant que canvia un valor que cap comanda no llegeix d'aquí: no canviaria res, i
    el registre de la passada el diria canviat.
    """
    found = {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("SMOKE")
    }
    if not smoke:
        return found
    # Els que `run/run.py` llegeix del que torna aquesta funció; la resta s'importen on
    # es fan servir.
    read = ("SEEDS", "TRIALS", "BEST", "EXPLAINED_DAYS")
    unread = [name for name in globals()[smoke] if name not in read]
    if unread:
        changed = (
            f"{unread[0]}, que cap comanda no llegeix d'una variant"
            if len(unread) == 1
            else f"{len(unread)} valors que cap comanda no llegeix d'una variant, el "
            f"primer {unread[0]}"
        )
        raise Refusal(
            f"{smoke} canvia {changed}; els que en llegeixen: {', '.join(read)}"
        )
    return found | globals()[smoke]
