"""Els números de la xarxa, que arriben de dos llocs.

**Els que tria la cerca no són aquí**: `search.py` els desa al disc, a
`paths.SEARCHES`, perquè els tria una cerca i no una persona. Aquesta família és
l'excepció a que els números d'un model siguin al seu fitxer, i el preu és que el
commit tot sol ja no fixa el que ha fet la predicció: qui ho fixa és el `run.json` de
l'execució, que els recull tots. Mentre no s'ha cercat sobre el conjunt de Lago et al.
(2021), `searched()` torna els que la cerca de l'article va triar per a un altre
mercat, copiats aquí i marcats.

Els que hi són són els que ningú no cerca, tots del codi de l'article: la finestra, la
part de validació, el lot, el sostre d'èpoques, la paciència, el retall del gradient i
la capa més petita que es manté; i el pressupost de la cerca.
"""
import json

from epf import paths
from epf.parameters import FOUR_YEARS

NAME = "dnn"

# Quatre anys de calibratge, com a Lago et al. (2021) i a Marcjasz et al. (2023).
WINDOW = FOUR_YEARS

# La part de la finestra que s'aparta per aturar l'entrenament, en files: el
# `percentage_val` del codi de Lago et al. (2021).
VALIDATION_SHARE = 0.25

# Xarxes que es mitjanen, cada una d'una cerca independent. L'article en fa quatre
# perquè una sola cerca és local; aquí n'hi ha una mentre es construeix, i pujar-ho
# multiplica el cost de la cerca per aquest número i no el del camí endavant.
MEMBERS = 1

# Mida del lot, sostre d'èpoques i paciència, els del codi de Lago et al. (2021) quan
# recalibra; el text de l'article no els compta entre els seus hiperparàmetres. El
# sostre hi és perquè qui talli sigui l'aturada.
BATCH = 192
EPOCHS = 1000
PATIENCE = 20

# El valor màxim de cada component del gradient, el `clipvalue` que el codi de Lago et
# al. (2021) posa al seu Adam.
CLIP = 10000

# Una capa amagada de menys neurones que aquestes no s'hi posa: el codi de Lago et al.
# (2021) en cerca de 25 a 400 a la segona capa i deixa fora les de sota 50, de manera
# que la cerca tria també si la xarxa té una capa o dues.
SMALLEST_LAYER = 50

# El pressupost de la cerca. Marcjasz et al. (2023) en fan 2048 amb Optuna i Lago et
# al. (2021) 1500 amb un estimador de Parzen; quants se'n fan aquí es decideix amb el
# cost d'una prova mesurat i no amb els seus.
TRIALS = 50

# Com es puntua una prova, que és l'esquema de Marcjasz et al. (2023) secció 4.2.1:
# tretze reajustos sobre lots de 28 dies, amb la finestra rodant 28 dies a cada pas.
# Tretze per vint-i-vuit són els 364 dies que es puntuen.
RECALIBRATIONS = 13
BATCH_DAYS = 28

# Els números mentre no se n'ha cercat cap sobre el conjunt de Lago et al. (2021): el
# millor de les 1500 proves de la cerca de l'article per al mercat alemany sense augment
# de dades, el fitxer `COPIED_FROM` del seu repositori, amb els noms d'aquí. Hi engega
# els onze blocs, que al conjunt d'aquí són els mateixos amb la demanda i l'eòlica més
# solar espanyoles. Es van triar amb les dades d'un altre mercat, de manera que cap
# xifra d'aquesta família no diu res fins que una cerca els substitueixi.
COPIED = {
    "blocks": ["day_of_week", "lag1", "lag2", "lag3", "lag7",
               "demand", "demand_lag1", "demand_lag7",
               "wind_solar", "wind_solar_lag1", "wind_solar_lag7"],
    "hidden": [458, 224],
    "activation": "sigmoid",
    "l1": 0.0,
    "dropout": 0.07089341603182506,
    "batch_normalisation": False,
    "initialiser": "he_uniform",
    "learning_rate": 0.0005497349000849452,
    "scale_inputs": "Std",
    "scale_outputs": "Median",
    "seed": 559,
}
COPIED_FROM = "DNN_hyperparameters_nl2_datDE_YT2_SF_CW4_1"
COPIED_TRIALS = 1500


def path(stage):
    """On viu el conjunt que ha triat la cerca sobre un conjunt d'informació."""
    return paths.SEARCHES / stage / f"{NAME}.json"


def searched():
    """Els números de la xarxa, d'on surten, i marcats com a sense cercar.

    Torna els valors **i la procedència** sota `search`: de quin fitxer i de quantes
    proves. Els valors tots sols dirien què s'ha fet servir i no si algú els havia
    triat de debò per a aquest mercat.
    """
    return COPIED | {"unsearched": True,
                     "search": {"copied_from": COPIED_FROM, "trials": COPIED_TRIALS}}


def save(stage, record):
    """Desa el que ha trobat una cerca sobre un conjunt, substituint el que hi hagi."""
    found = path(stage)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
