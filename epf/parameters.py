"""Tot el que s'ha triat per a una execució, un sol valor de cada.

El que és un fet del mercat o de la font no és aquí: això viu a `market.py`, a
`esios.py` i a `omie.py`, i canvia quan canvia el món, no quan es prova un número.
Els números d'un model tampoc: viuen amb el model, a
`models/<nom>/<nom>_hyperparameters.py`. Aquí només hi ha el que val per a
l'execució sencera.

Una execució és aquest fitxer en un commit, i el directori que en surt es diu pel
commit. Canviar qualsevol número d'aquí vol dir un commit nou, i per això el commit
identifica el resultat: un escombrat és un commit per punt i el missatge de cada un
diu què s'hi provava.
"""
import datetime

from epf.market import QUARTER_HOURLY_FROM

ONE_DAY = datetime.timedelta(days=1)

# ---------------------------------------------------------------- la partició

# Primer dia des del qual s'entrena. No és el primer dia que la font serveix, que
# és un fet i viu a `esios.py`: és fins on s'ha decidit arribar enrere, i es pot
# moure endavant sense tocar les dades.
HISTORY_START = datetime.date(2014, 1, 1)

# Dos anys de validació abans de la prova. S'hi trien els hiperparàmetres i s'hi
# comparen les famílies; els anys de prova no s'hi toquen mai.
VALIDATION = (datetime.date(2022, 10, 1), datetime.date(2024, 9, 30))

# Últim dia d'entrenament. Derivat i no escrit: és el dia abans de la validació, i
# escrit a part seria la mateixa frontera dues vegades.
TRAIN_END = VALIDATION[0] - ONE_DAY

# Dos anys de prova, avaluats per separat. El tall entre ells no es tria: és el dia
# que el mercat va passar a 96 productes quart-horaris, que és un fet i viu a
# `market.py`. Les dues vores de fora sí que es trien.
TEST_PERIODS = {
    "hourly": (datetime.date(2024, 10, 1), QUARTER_HOURLY_FROM - ONE_DAY),
    "quarter_hourly": (QUARTER_HOURLY_FROM, datetime.date(2026, 9, 30)),
}

# Els dos nivells d'execució. Una de prova és petita i ràpida a propòsit i el seu
# error no es desa, ni es cita, ni es compara enlloc; només una de mesura dóna un
# resultat. El nivell és al camí de l'execució, de manera que una xifra de prova no
# es pugui llegir mai com una mesura.
MEASUREMENT, TEST = "measurement", "test"

# Conjunt d'informació d'una execució, que va al camí. De moment només n'hi ha un:
# el preu i res més. L'etapa amb fonaments arriba amb les seves variables.
PRICE_ONLY = "price"

# ------------------------------------------------- el conjunt d'informació

# Retards de preu que admet l'etapa del preu, els del model de referència de Lago
# et al. (2021): fa 1, 2, 3 i 7 dies, les 24 hores de cada un. Totes les famílies
# els comparteixen, que és el que fa que una diferència entre elles sigui una
# diferència d'algorisme.
LAGS = (1, 2, 3, 7)

# Dies que es perden al principi de qualsevol tram: cap fila es pot formar fins
# que hi ha tots els retards.
WARM_UP = max(LAGS)

DAYS_PER_WEEK = 7

# ------------------------------------------------- la descripció de la part B

# El període que es descriu, del primer dia de les previsions D+1 al dia abans dels
# anys de prova, on acaba la validació de la part A. Són 300 setmanes de dimarts a
# dilluns, i totes les tries de la part B s'hi fan.
DEVELOPMENT = (datetime.date(2019, 1, 1), VALIDATION[1])

# Percentatge de les setmanes que va a entrenament, a validació i a prova, els de
# Trebbien et al. (2023).
SHARES = (48, 32, 20)

# Dies que surten de l'entrenament a cada costat d'una setmana retinguda. El preu del
# dia abans és una entrada, de manera que una setmana retinguda i el dia de cada costat
# es miren l'un a l'altre.
GAP = 1

# Una partició per llavor. La dispersió entre totes és la robustesa que es reporta.
SEEDS = tuple(range(10))

# Retards de preu dels dos conjunts de la part B: el dia abans, les seves 24 hores, que
# és d'on ha de venir el nivell.
PREVIOUS_DAY = (1,)

# ----------------------------------------------------------- aquesta execució

# Primer dia, últim dia i cada quants se'n pren un, per nivell. El de prova és un
# mes a propòsit, prou per comprovar la maquinària i prou curt per esperar-s'hi; el
# de mesura camina la validació sencera mostrejada, que cobreix les dues estacions i
# els dos règims. `every` no canvia res del que veu el model: cada dia mostrejat es
# prediu exactament igual que en una passada sencera, i només baixa la resolució de
# la comparació.
PERIODS = {
    MEASUREMENT: (VALIDATION[0], VALIDATION[1], 8),
    TEST: (datetime.date(2022, 6, 1), datetime.date(2022, 6, 30), 1),
}

# Llavor de tota família que en faci servir. És de l'execució i no d'un model, de
# manera que totes les que en prenen una prenen la mateixa.
SEED = 0
