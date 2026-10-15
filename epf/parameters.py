"""Tot el que s'ha triat per a una passada, un sol valor de cada.

El que és un fet del mercat o de la font no és aquí, llevat de les hores del dia: això
viu a `data/esios.py` i a `data/omie.py`, i canvia quan canvia el món, no quan es prova
un número.
Els números d'un model tampoc: viuen amb el model, a
`models/<nom>/<nom>_hyperparameters.py`. Aquí només hi ha el que val per a la
passada sencera.

Una passada és aquest fitxer en un commit, i el directori que en surt es diu pel
commit. Canviar qualsevol número d'aquí vol dir un commit nou, i per això el commit
identifica el resultat: un escombrat és un commit per punt i el missatge de cada un
diu què s'hi provava.
"""
import datetime

DAYS_PER_WEEK = 7

# Les hores d'un dia local i el nom de la columna de cada una. Són un fet i no una tria, i
# són aquí perquè les llegeix tot el paquet.
HOURS_PER_DAY = 24
HOUR_COLUMNS = [f"h{hour:02d}" for hour in range(HOURS_PER_DAY)]

# Els dos nivells d'una passada. Un assaig és petit i ràpid a propòsit i el seu error no
# es desa, ni es cita, ni es compara enlloc; només una mesura dóna un resultat. El nivell
# és al camí de la passada, de manera que una xifra d'assaig no es pugui llegir mai com
# una mesura. "Prova" no hi va: és el nom de les setmanes retingudes.
MEASUREMENT, SMOKE = "measurement", "smoke"

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

# Llavors i dies de prova que s'expliquen, per nivell. `None` vol dir tots. L'assaig és
# una llavor i 50 dies a propòsit, prou per passar tota la maquinària i prou curt per
# esperar-s'hi; la mesura és el protocol sencer.
DESCRIPTIONS = {
    MEASUREMENT: (SEEDS, None),
    SMOKE: ((0,), 50),
}

# Llavors i proves de la cerca de cada família, per nivell, i quantes de les millors se
# n'expliquen: Trebbien et al. (2023) en guarden les deu millors i no diuen quantes en
# proven, i cinquanta és el nombre del projecte. L'assaig és una llavor i dues proves,
# prou per passar tota la maquinària.
SEARCH_TRIALS = {
    MEASUREMENT: (SEEDS, 50),
    SMOKE: ((0,), 2),
}
BEST = 10

# Files de prova de cada llavor sobre les quals es calculen les interaccions de l'arbre,
# triades a l'atzar amb la llavor. Totes costarien unes onze vegades més.
INTERACTION_ROWS = 1000

# Llavor de tota família que en faci servir. És de la passada i no d'un model, de
# manera que totes les que en prenen una prenen la mateixa.
SEED = 0
