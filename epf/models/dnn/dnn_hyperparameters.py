"""Els números de la xarxa, que arriben de dos llocs.

**Els que tria la cerca no són aquí**: viuen al disc, a `paths.SEARCHES`, perquè els
tria `search.py` i no una persona, i els llegeix `searched()`. Aquesta família és
l'excepció a que els números d'un model siguin al seu fitxer, i el preu és que el
commit tot sol ja no fixa el que ha fet la predicció: qui ho fixa és el `run.json` de
l'execució, que els recull tots. Mentre no s'ha cercat sobre el conjunt de Lago et al.
(2021), en manlleva els d'una altra cerca, marcats; sense aquella cerca al disc, la
família no arrenca.

Els que hi són són els que ningú no cerca: la finestra i les setmanes de validació,
que són de l'article, i el pressupost de l'entrenament i de la cerca.
"""
import json

from epf import paths
from epf.parameters import WARM_UP

NAME = "dnn"

# Quatre anys de calibratge, que és el que fan tant Lago et al. (2021) com Marcjasz
# et al. (2023). Els seus "anys" són de 364 dies, 52 setmanes exactes que mantenen
# alineat el dia de la setmana, i el que es compta aquí són dies de calibratge un cop
# descomptat l'escalfament, igual que a LEAR.
WINDOW = 364 * 4 - WARM_UP

# Setmanes de la finestra que s'aparten per aturar l'entrenament quan deixa de
# millorar. De Lago et al. (2021).
VALIDATION_WEEKS = 42

# Xarxes que es mitjanen, cada una d'una cerca independent. L'article en fa quatre
# perquè una sola cerca és local; aquí n'hi ha una mentre es construeix, i pujar-ho
# multiplica el cost de la cerca per aquest número i no el del camí endavant.
MEMBERS = 1

# Mida del lot i sostre d'èpoques. Cap dels dos és dels articles: Lago et al. (2021)
# no els compta entre els seus hiperparàmetres i Marcjasz et al. (2023) no els diu.
# La paciència sí que és de la seva implementació en recalibrar, 20 èpoques sense
# millorar, i el sostre hi és perquè qui talli sigui l'aturada anticipada.
BATCH = 32
EPOCHS = 1000
PATIENCE = 20

# El pressupost de la cerca. Marcjasz et al. (2023) en fan 2048 amb Optuna i Lago et
# al. (2021) 1500 amb un estimador de Parzen; quants se'n fan aquí es decideix amb el
# cost d'una prova mesurat i no amb els seus.
TRIALS = 50

# Com es puntua una prova, que és l'esquema de Marcjasz et al. (2023) secció 4.2.1:
# tretze reajustos sobre lots de 28 dies, amb la finestra rodant 28 dies a cada pas.
# Tretze per vint-i-vuit són els 364 dies que es puntuen.
RECALIBRATIONS = 13
BATCH_DAYS = 28


# La cerca d'on surten els números mentre no se n'ha fet cap sobre el conjunt de Lago
# et al. (2021): la del conjunt de només preu, ja retirat. Tria retards de preu i el dia
# de la setmana i cap previsió, de manera que cap xifra d'aquesta família no diu res
# fins que la cerca sobre aquell conjunt la substitueixi.
BORROWED = "price"


def path(stage):
    """On viu el conjunt que ha triat la cerca sobre un conjunt d'informació."""
    return paths.SEARCHES / stage / f"{NAME}.json"


def searched():
    """Els números de la cerca manllevada, d'on surten, i marcats com a sense cercar.

    Refusa si no hi és: aquesta família no té valors per defecte, i arrencar-la amb uns
    d'inventats donaria un error que no diu res de la xarxa de l'article.

    Torna els valors **i la procedència** sota `search`: quantes proves, quins dies,
    sobre quin conjunt i quin error. Els valors tots sols dirien què s'ha fet servir i no
    si algú els havia triat de debò, i una cerca de vint proves i una de mil deixen el
    mateix rastre al `run.json` si no s'hi desa el número.
    """
    found = path(BORROWED)
    if not found.exists():
        raise FileNotFoundError(
            f"no hi ha a {found} la cerca del conjunt de només preu, que la família "
            f"{NAME} fa servir mentre no se n'hagi fet cap sobre el de Lago et al. (2021)")
    record = json.loads(found.read_text())
    return record["hyperparameters"] | {
        "unsearched": True,
        "search": {name: value for name, value in record.items()
                   if name != "hyperparameters"}}


def save(stage, record):
    """Desa el que ha trobat una cerca sobre un conjunt, substituint el que hi hagi."""
    found = path(stage)
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(json.dumps(record, indent=2) + "\n")
