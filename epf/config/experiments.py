"""Tot el que s'ha triat: la partició, el conjunt d'informació i les configuracions.

El que és un fet del mercat o de la font no és aquí: això viu a `market.py`, a
`esios.py` i a `omie.py`, i canvia quan canvia el món, no quan es prova un número.
Aquí només hi ha el que algú ha decidit i pot tornar a decidir.

Una execució és una configuració d'aquest fitxer en un commit, i el directori que
en surt es diu `<commit>_<configuració>`. Canviar qualsevol número d'aquí vol dir
un commit nou, i per això el commit identifica el resultat: el missatge del commit
és el nom llegible de l'execució.
"""
import datetime

from epf.config.market import QUARTER_HOURLY_FROM

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

# Finestres de calibratge de LEAR, en dies de calibratge. Els "anys" de l'article
# són de 364 dies, 52 setmanes exactes que mantenen alineat el dia de la setmana;
# les seves finestres de 1092 i 1456 dies de preus donen aquests dies un cop
# descomptat l'escalfament. None vol dir tot l'historial, que l'article no fa
# servir i que hi és per posar a prova la decisió d'arribar fins al 2014.
EIGHT_WEEKS = 8 * DAYS_PER_WEEK - WARM_UP
TWELVE_WEEKS = 12 * DAYS_PER_WEEK - WARM_UP
LEAR_WINDOWS = (EIGHT_WEEKS, TWELVE_WEEKS, 364 * 3 - WARM_UP, 364 * 4 - WARM_UP, None)

# Retards que porta cada finestra curta. Amb els quatre, la matriu de disseny té
# 103 columnes i LassoLarsIC demana dues files més que columnes, que cap de les
# dues finestres curtes no té: cada una porta els que hi caben. Si es canvien les
# finestres i la parella queda incoherent, LEAR falla en comptar les files.
SHORT_LAGS = {EIGHT_WEEKS: (1,), TWELVE_WEEKS: (1, 2)}

# ------------------------------------------------------- les configuracions

BASE = dict(
    # Primer dia, últim dia i cada quants se'n pren un, per nivell. El de prova és
    # un mes a propòsit, prou per comprovar la maquinària i prou curt per esperar-
    # s'hi; el de mesura camina la validació sencera mostrejada, que cobreix les dues
    # estacions i els dos règims. `every` no canvia res del que veu el model: cada
    # dia mostrejat es prediu exactament igual que en una passada sencera, i només
    # baixa la resolució de la comparació.
    periods={
        MEASUREMENT: (VALIDATION[0], VALIDATION[1], 8),
        TEST: (datetime.date(2022, 6, 1), datetime.date(2022, 6, 30), 1),
    },

    # LEAR. El sostre d'iteracions del descens de coordenades és 10000 i no 2500:
    # mesurat el 2026-10-01 sobre el 2022-09-01, a 2500 la finestra de 1085 dies
    # deixava 13 dels seus 24 ajusts per hora sense convergir i la de 1449 en
    # deixava 10, i la predicció d'aquell dia es movia fins a 0,566 EUR/MWh. A
    # 50000 i a 200000 surt bit a bit igual que a 10000.
    lear_windows=LEAR_WINDOWS,
    lear_max_iter=10000,

    # LightGBM. Punt de partida i no els valors per defecte de la llibreria, que
    # arrenca en 100 arbres i taxa 0,1.
    window=1085,
    trees=500,
    rate=0.05,
    leaves=31,

    # MLP. Dues capes amagades, que és l'estructura de la xarxa de l'article un
    # cop descomptades l'entrada i la sortida, i quatre xarxes mitjanades, que és
    # el que fa l'article perquè una xarxa és sensible a la llavor. El sostre
    # d'èpoques és el de scikit-learn: mesurat sobre sis finestres reals de 1085 i
    # 1449 dies i dues llavors, l'aturada anticipada salta entre l'època 33 i la
    # 126, de manera que 200 no talla cap ajust.
    mlp_window=1085,
    mlp_hidden=(128, 128),
    mlp_members=4,
    mlp_alpha=1e-4,
    mlp_learning_rate=1e-3,
    mlp_validation_fraction=0.1,
    mlp_no_change=10,
    mlp_max_iter=200,

    # Llavor de tota la família que en faci servir.
    seed=0,
)

# Les configuracions que s'executen. Una entrada nova és un commit nou, i el seu
# missatge és el que després explica el directori.
CONFIGURATIONS = {
    "base": BASE,
}


def configuration(name):
    """La configuració amb aquest nom, o un error que diu quines hi ha."""
    if name not in CONFIGURATIONS:
        raise KeyError(f"cap configuració anomenada {name!r}; "
                       f"n'hi ha {sorted(CONFIGURATIONS)}")
    return CONFIGURATIONS[name]
