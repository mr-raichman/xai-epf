"""Els valors que tria una passada sencera, un de cada.

Són els d'una mesura, i un assaig en canvia els de la seva variant. Una passada els desa
tots al registre de cada família.
"""

import datetime

DAYS_PER_WEEK = 7

# Les hores d'un dia local i el nom de la columna de cada una.
HOURS_PER_DAY = 24
HOUR_COLUMNS = [f"h{hour:02d}" for hour in range(HOURS_PER_DAY)]

# ------------------------------------------------------------ la descripció

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

# Una partició per llavor.
SEEDS = tuple(range(10))

# Retards de preu del conjunt d'informació: el dia abans, les seves 24 hores.
PREVIOUS_DAY = (1,)

# El dia de cada sèrie d'entrada, en dies abans del lliurament: les previsions D+1, del
# mateix dia que es lliura, i el programa del dia abans, tancat dos dies abans i a
# l'abast de tothom abans del tancament de la subhasta.
FORECAST_LAG = 0
SCHEDULE_LAG = 1

# Dies de prova que s'expliquen de cada llavor, comptats des del primer. `None` vol dir
# tots, el protocol sencer.
EXPLAINED_DAYS = None

# Intents de la cerca de cada família a cada llavor, i quants dels millors se
# n'expliquen: Trebbien et al. (2023) en guarden els deu millors i no diuen quants en
# proven, i cinquanta és el nombre del projecte.
TRIALS = 50
BEST = 10

# Files explicades de cada llavor sobre les quals es calculen les interaccions de
# l'arbre, triades a l'atzar amb la llavor.
INTERACTION_ROWS = 1000

# Llavor de tota família que en faci servir.
SEED = 0

# ------------------------------------------------------------ els processos

# Quants processos fan alhora els ajustos d'una cerca o d'una passada.
WORKERS = 4

# Els fils de cada procés, sempre els mateixos: amb uns altres un ajust dona uns altres
# números, i amb aquests una passada dona els mateixos siguin quants siguin els
# processos.
THREADS = 4

# ------------------------------------------------------------ l'assaig

# El que `--smoke` posa en lloc dels valors de la mesura, i res més, en dues variants:
# `SMOKE`, i `SMOKE_1`, que no fixa els dies explicats i n'explica tots els de prova. La
# que fa servir un assaig la tria `chosen`, i es canvia editant-la.
SMOKE = {"SEEDS": (0,), "EXPLAINED_DAYS": 50, "TRIALS": 2}
SMOKE_1 = {"SEEDS": tuple(range(5)), "TRIALS": 10}


def chosen(smoke):
    """Torna cada valor d'aquest fitxer pel seu nom, els d'assaig si `smoke`."""
    found = {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("SMOKE")
    }
    return (found | SMOKE_1) if smoke else found
