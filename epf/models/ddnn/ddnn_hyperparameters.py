"""Els números de les dues xarxes distribucionals.

Com la xarxa de punt, els que tria una cerca no són aquí sinó al disc. **I encara no
s'han cercat**: mentre la cerca està ajornada, aquestes famílies prenen els que fa servir
la xarxa de punt, que de moment també són manllevats.

El sostre d'aquesta simplificació, i és gros: aquells números es van triar minimitzant
l'error absolut d'una predicció puntual, i aquestes xarxes minimitzen una
log-versemblança. No hi ha cap raó per esperar que els mateixos siguin bons per a les
dues coses, de manera que **cap xifra d'aquestes famílies no diu res mentre no tinguin
la seva pròpia cerca**. El que ho aixeca és una entrada per família a `searches/`.

L'L1 de la capa de distribució, que l'article cerca per separat per a cada paràmetre,
és zero mentre no se cerqui: l'article en té `2P` hiperparàmetres i aquí no n'hi ha cap
de triat.
"""
from epf.models.dnn import dnn_hyperparameters

# Els mateixos que la xarxa de punt, perquè l'article també hi comparteix la finestra,
# les setmanes de validació i la partició: les dues xarxes "differ only in the output
# construction in the network and hyperparameters corresponding to the missing
# distribution layer".
WINDOW = dnn_hyperparameters.WINDOW
VALIDATION_WEEKS = dnn_hyperparameters.VALIDATION_WEEKS
MEMBERS = dnn_hyperparameters.MEMBERS
BATCH = dnn_hyperparameters.BATCH
EPOCHS = dnn_hyperparameters.EPOCHS
PATIENCE = dnn_hyperparameters.PATIENCE


def searched():
    """Els números que fa servir una xarxa distribucional, amb d'on surten.

    De moment els de la xarxa de punt, marcats com a manllevats, de manera que el
    `run.json` de qualsevol execució d'aquestes famílies diu que els seus números es
    van triar per a una altra pèrdua.
    """
    borrowed = dnn_hyperparameters.searched()
    return borrowed | {"search": borrowed["search"] | {"borrowed_from": "dnn"}}
