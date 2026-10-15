"""Tot el que s'ha triat per a una passada, un sol valor de cada.

El que és un fet del mercat o de la font no és aquí, llevat de les hores del dia: això
viu a `dataset/esios.py`, i canvia quan canvia el món, no quan es prova un número.
Els números d'un model tampoc: viuen amb el model, a
`models/<model>/hyperparameters.py`. Aquí només hi ha el que val per a la
passada sencera.

Són els valors d'una mesura; `--smoke` posa els de `SMOKE`, al final, en lloc dels seus, i
no en canvia cap altre. Una passada els desa tots al registre de cada família, amb
l'empremta del codi i la del conjunt de dades que la van fer, de manera que cada punt
d'un escombrat es llegeix del seu registre.
"""
import datetime

DAYS_PER_WEEK = 7

# Les hores d'un dia local i el nom de la columna de cada una. Són un fet i no una tria, i
# són aquí perquè les llegeix tot el paquet.
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

# Percentatge de les setmanes que va a entrenament, a validació i a prova, els de
# Trebbien et al. (2023).
SHARES = (48, 32, 20)

# Dies que surten de l'entrenament a cada costat d'una setmana retinguda. El preu del
# dia abans és una entrada, de manera que una setmana retinguda i el dia de cada costat
# es miren l'un a l'altre.
GAP = 1

# Una partició per llavor. La dispersió entre totes és la robustesa que es reporta.
SEEDS = tuple(range(10))

# Retards de preu del conjunt: el dia abans, les seves 24 hores, que és d'on ha de venir
# el nivell.
PREVIOUS_DAY = (1,)

# El dia de cada sèrie d'entrada, en dies abans del lliurament: les previsions D+1, del
# mateix dia que es lliura, i el programa del dia abans, tancat dos dies abans i a l'abast
# de tothom abans del tancament de la subhasta.
FORECAST_LAG = 0
SCHEDULE_LAG = 1

# Dies de prova que s'expliquen de cada llavor, comptats des del primer. `None` vol dir
# tots, el protocol sencer.
EXPLAINED_DAYS = None

# Proves de la cerca de cada família a cada llavor, i quantes de les millors se
# n'expliquen: Trebbien et al. (2023) en guarden les deu millors i no diuen quantes en
# proven, i cinquanta és el nombre del projecte.
TRIALS = 50
BEST = 10

# Files de prova de cada llavor sobre les quals es calculen les interaccions de l'arbre,
# triades a l'atzar amb la llavor. Totes costarien unes onze vegades més.
INTERACTION_ROWS = 1000

# Llavor de tota família que en faci servir. És de la passada i no d'un model, de
# manera que totes les que en prenen una prenen la mateixa.
SEED = 0

# ------------------------------------------------------------ l'assaig

# El que `--smoke` posa en lloc dels valors de la mesura, i res més. Un assaig és petit i
# ràpid a propòsit, una llavor, 50 dies explicats i dues proves per cerca, prou per passar
# tota la maquinària i prou curt per esperar-s'hi. El seu error no es desa, ni es cita,
# ni es compara enlloc; només una mesura dóna un resultat, i el nom d'una passada o d'una
# cerca d'assaig comença amb `assaig_`, de manera que una xifra d'assaig no es pugui llegir
# mai com una mesura. "Prova" no hi va: és el nom de les setmanes retingudes.
SMOKE = {"SEEDS": (0,), "EXPLAINED_DAYS": 50, "TRIALS": 2}


def chosen(smoke):
    """Cada valor d'aquest fitxer pel seu nom, amb els de `SMOKE` en lloc dels de la mesura
    si és un assaig. `SMOKE` mateix no hi és: són els valors que fa servir una passada."""
    found = {name: value for name, value in globals().items()
             if name.isupper() and name != "SMOKE"}
    return (found | SMOKE) if smoke else found
