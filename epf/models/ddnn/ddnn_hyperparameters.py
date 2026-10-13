"""Els números de les dues xarxes distribucionals.

Com la xarxa de punt, els que tria una cerca no són aquí sinó al disc. **I encara no
s'han cercat**: mentre la cerca està ajornada, aquestes famílies prenen els que va
triar la cerca de la xarxa de punt sobre el conjunt de només preu, ja retirat, llegits
del seu fitxer. La xarxa de punt segueix ara el codi de Lago et al. (2021), amb escalats,
una sola activació i una llavor que la de Marcjasz et al. (2023) no té, i aquestes es
queden amb la seva xarxa i amb el seu entrenament.

El sostre d'aquesta simplificació, i és gros: aquells números es van triar minimitzant
l'error absolut d'una predicció puntual, i aquestes xarxes minimitzen una
log-versemblança. No hi ha cap raó per esperar que els mateixos siguin bons per a les
dues coses, de manera que **cap xifra d'aquestes famílies no diu res mentre no tinguin
la seva pròpia cerca**. El que ho aixeca és una entrada per família a `searches/`.

L'L1 de la capa de distribució, que l'article cerca per separat per a cada paràmetre,
és zero mentre no se cerqui: l'article en té `2P` hiperparàmetres i aquí no n'hi ha cap
de triat.
"""
import json

from epf import paths
from epf.models.dnn import dnn_hyperparameters

# La finestra, el sostre d'èpoques i la paciència, els de la xarxa de punt, perquè
# l'article les comparteix: les dues xarxes "differ only in the output construction in
# the network and hyperparameters corresponding to the missing distribution layer".
WINDOW = dnn_hyperparameters.WINDOW
EPOCHS = dnn_hyperparameters.EPOCHS
PATIENCE = dnn_hyperparameters.PATIENCE

# Setmanes de la finestra que s'aparten per aturar l'entrenament, de Lago et al. (2021)
# segons el seu text, i la mida del lot. Les de la xarxa de punt abans que seguís el codi
# de l'article, que aquestes conserven mentre no tinguin la seva cerca.
VALIDATION_WEEKS = 42
BATCH = 32

# Una xarxa per distribució, on Marcjasz et al. (2023) en mitjanen quatre: cada membre
# és una cerca més, i el que costa és la cerca.
MEMBERS = 1

# La cerca d'on surten els números: la de la xarxa de punt sobre el conjunt de només
# preu, que tria retards de preu i el dia de la setmana i cap previsió.
BORROWED = "price"


def borrowed():
    """On és el fitxer d'aquella cerca, que era d'una sola xarxa i no d'un membre."""
    return paths.SEARCHES / BORROWED / f"{dnn_hyperparameters.NAME}.json"


def searched():
    """Els números que fa servir una xarxa distribucional, amb d'on surten.

    Els de la cerca de la xarxa de punt sobre el conjunt de només preu, marcats com a
    sense cercar i manllevats, de manera que el `run.json` de qualsevol execució
    d'aquestes famílies diu que els seus números es van triar per a una altra pèrdua.
    Refusa si aquella cerca no és al disc: aquestes famílies no tenen valors per
    defecte, i arrencar-les amb uns d'inventats donaria un error que no diu res de la
    xarxa de l'article.
    """
    found = borrowed()
    if not found.exists():
        raise FileNotFoundError(
            f"no hi ha a {found} la cerca de la xarxa de punt sobre el conjunt de només "
            "preu, que les xarxes distribucionals fan servir mentre no tinguin la seva")
    record = json.loads(found.read_text())
    provenance = {name: value for name, value in record.items()
                  if name != "hyperparameters"}
    return record["hyperparameters"] | {
        "unsearched": True, "search": provenance | {"borrowed_from": "dnn"}}
