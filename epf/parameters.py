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

# Percentatges de les setmanes que van a entrenament, a validació i a prova, els de
# Trebbien et al. (2023).
SHARES = (48, 32, 20)

# Una partició per seed: el doble de les deu de Trebbien et al. (2023), perquè les seves
# setmanes de prova cobreixin gairebé tot el període.
SEEDS = tuple(range(20))


### Ajust i cerca ###

# La seed d'on surt cada tria a l'atzar del projecte, per `seeded`.
PROJECT_SEED = 0

# Trials de la cerca de cada família a cada seed: amb `None`, els seus, tota la graella
# de l'arbre i de l'elastic net i 100 de la xarxa; un nombre, el d'un assaig, en limita
# tots.
TRIALS = None

# Les millors configuracions diferents de cada seed que s'expliquen: Trebbien et al.
# (2023) en guarden les deu millors.
BEST = 10


### Explicació ###

# Dies de prova que s'expliquen de cada seed, comptats des del primer. `None` vol dir
# tots, el protocol sencer.
EXPLAINED_DAYS = None

# Files explicades de cada seed sobre les quals es calculen les interaccions de l'arbre,
# triades a l'atzar amb la seed: amb 2000, els tres socis més forts de cada sèrie són
# els de totes les files en el 96.4 % de les mostres o més, en una seed.
INTERACTION_ROWS = 2000


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
# `--smoke` o `--smoke1`. Les seeds de cada una són de les que fan barats els primers
# trials de la xarxa neuronal, mentre no en canviïn l'espai ni la seed del sampler, i
# les de les tres no es repeteixen, perquè una passada amplia qualsevol study d'assaig
# que troba.

# La més ràpida, per comparar dues passades byte a byte i veure que el codi no falla:
# dues seeds de dues configuracions, el mínim que fa cada número entre seeds i dins
# d'una.
SMOKE = {"SEEDS": (3, 11), "TRIALS": 2, "EXPLAINED_DAYS": 7}
# Prou per jutjar si una figura val la pena.
SMOKE_1 = {"SEEDS": (2, 16, 18), "TRIALS": 6, "BEST": 3, "EXPLAINED_DAYS": 30}
# Prou per veure la direcció i les relacions evidents, sense ser una mesura.
SMOKE_2 = {"SEEDS": (1, 4, 17), "TRIALS": 20, "BEST": 5}


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


def flag(variant):
    """Torna l'opció que tria una variant d'assaig: `--smoke1` per `SMOKE_1`."""
    return f"--{variant.lower().replace('_', '')}"


def chosen(smoke, read):
    """Torna cada valor d'aquest fitxer pel seu nom, els de la variant d'assaig `smoke`.

    `smoke` és el nom d'una variant, `SMOKE_1`, o fals per a una mesura, i `read` els
    noms que qui la crida llegeix del que torna. Refusa una variant que no hi és, i una
    que canvia un altre valor: no canviaria res, i el registre de la passada el diria
    canviat.
    """
    found = {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("SMOKE")
    }
    if not smoke:
        return found
    if smoke not in globals():
        raise Refusal(f"no existeix {smoke}")
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
