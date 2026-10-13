"""Els números de les dues xarxes distribucionals, que arriben de dos llocs.

Com els de la xarxa de punt, els que tria una cerca no són aquí sinó al disc, un fitxer
per família que desa `search.py`, i `searched(name)` els llegeix. **I encara no s'han cercat**: mentre no,
cada família fa servir el millor conjunt de les cerques que Marcjasz et al. (2023) van
publicar per al mercat alemany, copiat aquí i marcat. Es van triar amb les dades d'un
altre mercat, de manera que **cap xifra d'aquestes famílies no diu res fins que tinguin
la seva pròpia cerca**.

Els que hi són són els que ningú no cerca, tots del codi de l'article: la finestra, el
lot, el sostre d'èpoques, la paciència i la part de dies que s'aparta per aturar-se.
"""
import json

from epf import paths
from epf.models.dnn import dnn_hyperparameters
from epf.parameters import INPUTS

# Quatre anys de calibratge, els de la xarxa de punt: el codi de l'article ajusta cada
# dia amb els últims 1456 dies menys els set de l'escalfament.
WINDOW = dnn_hyperparameters.WINDOW

# Mida del lot, sostre d'èpoques i paciència de l'aturada, i la part dels dies de la
# finestra que s'aparta a l'atzar per aturar-se, dies solts: els del codi de l'article.
BATCH = 32
EPOCHS = 1500
PATIENCE = 50
HELD_OUT = 0.2

# Una xarxa per distribució, on Marcjasz et al. (2023) en mitjanen quatre: cada membre
# és una cerca més, i el que costa és la cerca.
MEMBERS = 1

# Els números de cada família mentre no s'ha cercat: de les quatre cerques de 2048 proves
# que l'article va publicar per distribució, la de millor puntuació pròpia, el fitxer
# `COPIED_FROM` del seu repositori, amb els noms d'aquí. Els blocs dels combustibles no hi
# són, perquè el conjunt d'aquí no en porta, i el de l'eòlica més solar de fa una setmana
# hi és apagat, perquè la seva cerca no el té. L'L1 dels paràmetres va en l'ordre de
# `distributions`, i no en el del codi de l'article, que posa el pes de la cua de Johnson
# SU abans de l'asimetria.
COPIED = {
    "ddnn_jsu": {
        "blocks": ["day_of_week", "lag1", "demand", "demand_lag7", "wind_solar",
                   "wind_solar_lag1"],
        "hidden": [243, 895], "activation": ["softplus", "elu"], "dropout": 0.0,
        "l1": [0.0, 0.0], "activity_l1": [0.0, 0.0], "parameter_l1": [0.0, 0.0, 0.0, 0.0],
        "learning_rate": 0.000281493495779889},
    "ddnn_normal": {
        "blocks": ["lag1", "lag2", "lag7", "demand", "demand_lag7", "wind_solar",
                   "wind_solar_lag1"],
        "hidden": [110, 823], "activation": ["softplus", "relu"], "dropout": 0.0,
        "l1": [0.0, 0.0], "activity_l1": [1.0076687725977794e-05, 0.0],
        "parameter_l1": [0.0, 0.0], "learning_rate": 0.00032041091802416773},
}
COPIED_FROM = {"ddnn_jsu": "FINAL_DE_selection_prob_jsu3",
               "ddnn_normal": "FINAL_DE_selection_prob_normal4"}
COPIED_TRIALS = 2048


def path(stage, name):
    """On viu el que ha triat la cerca d'una família sobre un conjunt d'informació."""
    return paths.SEARCHES / stage / f"{name}.json"


def searched(name):
    """Els números de la família `name` i d'on surten.

    Els de la seva cerca sobre el conjunt que es camina, si n'hi ha; si no, els copiats,
    marcats com a sense cercar, amb el fitxer d'on surten. El conjunt que es camina és un
    de sol, i el desempaquetat peta el dia que n'hi hagi dos.
    """
    (stage,) = INPUTS
    found = path(stage, name)
    if not found.exists():
        return COPIED[name] | {"unsearched": True,
                               "search": {"copied_from": COPIED_FROM[name],
                                          "trials": COPIED_TRIALS}}
    record = json.loads(found.read_text())
    return record["hyperparameters"] | {
        "search": {key: value for key, value in record.items() if key != "hyperparameters"}}


def save(stage, name, record):
    """Desa el que ha trobat la cerca de la família `name` sobre un conjunt, substituint el
    que hi hagi."""
    found = path(stage, name)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
