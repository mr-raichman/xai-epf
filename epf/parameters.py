"""Tot el que s'ha triat per a una passada, un sol valor de cada.

El que és un fet del mercat o de la font no és aquí: això viu a `market.py`, a
`esios.py` i a `omie.py`, i canvia quan canvia el món, no quan es prova un número.
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

# Els dos nivells d'una passada. Un assaig és petit i ràpid a propòsit i el seu error no
# es desa, ni es cita, ni es compara enlloc; només una mesura dóna un resultat. El nivell
# és al camí de la passada, de manera que una xifra d'assaig no es pugui llegir mai com
# una mesura. "Prova" no hi va: és el nom de les setmanes retingudes.
MEASUREMENT, SMOKE = "measurement", "smoke"

# ------------------------------------------------------------ la descripció

# El període que es descriu, del primer dia de les previsions D+1 fins al 2024-09-30.
# Són 300 setmanes de dimarts a dilluns, i totes les tries s'hi fan.
DEVELOPMENT = (datetime.date(2019, 1, 1), datetime.date(2024, 9, 30))

# Percentatge de les setmanes que va a entrenament, a validació i a prova, els de
# Trebbien et al. (2023).
SHARES = (48, 32, 20)

# Dies que surten de l'entrenament a cada costat d'una setmana retinguda. El preu del
# dia abans és una entrada, de manera que una setmana retinguda i el dia de cada costat
# es miren l'un a l'altre.
GAP = 1

# Una partició per llavor. La dispersió entre totes és la robustesa que es reporta.
SEEDS = tuple(range(10))

# Retards de preu dels conjunts: el dia abans, les seves 24 hores, que és d'on ha de
# venir el nivell.
PREVIOUS_DAY = (1,)

# Llavors i dies de prova que s'expliquen, per nivell. `None` vol dir tots. L'assaig és
# una llavor i 50 dies a propòsit, prou per passar tota la maquinària i prou curt per
# esperar-s'hi; la mesura és el protocol sencer.
DESCRIPTIONS = {
    MEASUREMENT: (SEEDS, None),
    SMOKE: ((0,), 50),
}

# Files de prova de cada llavor sobre les quals es calculen les interaccions de l'arbre,
# triades a l'atzar amb la llavor. Totes costarien unes onze vegades més.
INTERACTION_ROWS = 1000

# Llavor de tota família que en faci servir. És de la passada i no d'un model, de
# manera que totes les que en prenen una prenen la mateixa.
SEED = 0
