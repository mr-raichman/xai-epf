"""Els números de la xarxa, que arriben de dos llocs.

**Els que tria la cerca no són aquí**: `search.py` els desa al disc, a
`paths.SEARCHES`, perquè els tria una cerca i no una persona. Aquesta família és
l'excepció a que els números d'un model siguin al seu fitxer, i el preu és que el
commit tot sol ja no fixa el que ha fet la predicció: qui ho fixa és el `run.json` de
l'execució, que els recull tots. Cada membre té els seus, de la seva cerca. Mentre no
s'ha cercat sobre el conjunt de Lago et al. (2021), `searched()` torna per a cada membre
els que una de les cerques de l'article va triar per a un altre mercat, copiats aquí i
marcats; un cop s'hi ha cercat, els de la cerca.

Els que hi són són els que ningú no cerca, tots del codi de l'article: la finestra, la
part de validació, el lot, el sostre d'èpoques, la paciència, el retall del gradient i
la capa més petita que es manté; i el pressupost de la cerca.
"""
import json

from epf import paths
from epf.parameters import FOUR_YEARS, INPUTS

NAME = "dnn"

# Quatre anys de calibratge, com a Lago et al. (2021) i a Marcjasz et al. (2023).
WINDOW = FOUR_YEARS

# La part de la finestra que s'aparta per aturar l'entrenament, en files: el
# `percentage_val` del codi de Lago et al. (2021).
VALIDATION_SHARE = 0.25

# Xarxes que es mitjanen, cada una de la seva cerca: les quatre de Lago et al. (2021),
# que fa la cerca quatre vegades amb una llavor inicial diferent perquè cada una troba
# un mínim local, i cap no és clarament millor que les altres.
MEMBERS = 4

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

# El pressupost de la cerca de cada membre, les proves de Lago et al. (2021).
TRIALS = 1500

# Els números de cada membre mentre no se n'ha cercat cap sobre el conjunt de Lago et al.
# (2021): el millor de les 1500 proves de cada una de les quatre cerques que l'article
# va publicar per al mercat alemany, els fitxers `COPIED_FROM` del seu repositori, amb
# els noms d'aquí. La primera previsió d'aquell conjunt és la de càrrega i la segona la
# d'eòlica més solar, com aquí `demand` i `wind_solar`. Dues de les quatre cerques es van
# fer amb un augment de dades que aquest projecte no té. Es van triar amb les dades d'un
# altre mercat, de manera que cap xifra d'aquesta família no diu res fins que les
# cerques els substitueixin.
COPIED = (
    {"blocks": ["day_of_week", "lag1", "lag2", "lag3", "lag7",
                "demand", "demand_lag1", "demand_lag7",
                "wind_solar", "wind_solar_lag1", "wind_solar_lag7"],
     "hidden": [458, 224], "activation": "sigmoid", "l1": 0.0,
     "dropout": 0.07089341603182506, "batch_normalisation": False,
     "initialiser": "he_uniform", "learning_rate": 0.0005497349000849452,
     "scale_inputs": "Std", "scale_outputs": "Median", "seed": 559},
    {"blocks": ["lag1", "lag2", "lag3", "lag7",
                "demand", "demand_lag1", "demand_lag7",
                "wind_solar", "wind_solar_lag1", "wind_solar_lag7"],
     "hidden": [476, 273], "activation": "PReLU", "l1": 0.0,
     "dropout": 0.12771044614422908, "batch_normalisation": True,
     "initialiser": "he_uniform", "learning_rate": 0.0013245099370788327,
     "scale_inputs": "Median", "scale_outputs": "Invariant", "seed": 979},
    {"blocks": ["day_of_week", "lag1", "lag2", "lag3", "lag7",
                "demand", "demand_lag1", "demand_lag7",
                "wind_solar", "wind_solar_lag1"],
     "hidden": [329, 379], "activation": "softplus", "l1": 0.0,
     "dropout": 0.4553221835706901, "batch_normalisation": True,
     "initialiser": "glorot_uniform", "learning_rate": 0.00670092151052075,
     "scale_inputs": "Std", "scale_outputs": "Median", "seed": 199},
    {"blocks": ["day_of_week", "lag1", "lag2", "lag7",
                "demand", "demand_lag1", "demand_lag7",
                "wind_solar", "wind_solar_lag1"],
     "hidden": [457, 257], "activation": "PReLU", "l1": 0.0,
     "dropout": 0.25094976597737745, "batch_normalisation": True,
     "initialiser": "he_normal", "learning_rate": 0.0016702693615958724,
     "scale_inputs": "Std", "scale_outputs": "Std", "seed": 933},
)
COPIED_FROM = tuple(f"DNN_hyperparameters_nl2_datDE_YT2_{variant}_CW4_1"
                    for variant in ("SF", "SF_DA", "SF_SF", "SF_SF_DA"))
COPIED_TRIALS = 1500


def path(stage, member):
    """On viu el conjunt que ha triat la cerca d'un membre sobre un conjunt
    d'informació."""
    return paths.SEARCHES / stage / f"{NAME}_{member}.json"


def _member(stage, member):
    """Els números d'un membre i d'on surten: els de la seva cerca si n'hi ha, i si no
    els copiats, marcats com a sense cercar."""
    found = path(stage, member)
    if not found.exists():
        return COPIED[member] | {
            "unsearched": True,
            "search": {"copied_from": COPIED_FROM[member], "trials": COPIED_TRIALS}}
    record = json.loads(found.read_text())
    return record["hyperparameters"] | {
        "search": {name: value for name, value in record.items()
                   if name != "hyperparameters"}}


def searched():
    """Els números de cada membre, en ordre sota `members`, i d'on surten.

    Cada un amb la seva procedència sota `search`: quantes proves, i quins dies o de quin
    fitxer. Els valors tots sols dirien què s'ha fet servir i no si algú els havia triat
    de debò per a aquest mercat, i una cerca de vint proves i una de mil deixarien el
    mateix rastre al `run.json`. Mentre algun membre no s'ha cercat, `unsearched` hi és
    també a dalt, que és on el porten totes les famílies.

    El conjunt que es camina és un de sol, i el desempaquetat peta el dia que n'hi hagi
    dos, que és quan aquesta funció haurà de saber quin.
    """
    (stage,) = INPUTS
    members = [_member(stage, member) for member in range(MEMBERS)]
    found = {"members": members}
    if any("unsearched" in member for member in members):
        found["unsearched"] = True
    return found


def save(stage, member, record):
    """Desa el que ha trobat la cerca d'un membre sobre un conjunt, substituint el que hi
    hagi."""
    found = path(stage, member)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
