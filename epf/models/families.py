"""Quines famílies hi ha, quina matriu pren cada una i amb quines funcions es crida.

La taula que el camí endavant recorre i que la passada d'atribució torna a llegir per
saber com s'explica un pronòstic. Viu aquí i no al corredor perquè la necessiten tots
dos (`architecture.md` 8.3), i perquè què és una família és dels models i no de qui els
camina.

Cap número de cap model hi és lligat: cada un els pren per defecte del seu propi
`models/<nom>/<nom>_hyperparameters.py`, de manera que una prova en pot passar uns de
més petits sense que el mòdul hagi de saber que hi ha proves.
"""
import collections
import importlib

from epf.models.boosting import boosting
from epf.models.ddnn import ddnn
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.dnn import dnn
from epf.models.lear import lear
from epf.models.mlp import mlp
from epf.models.naive import naive

DAILY, HOURLY = "daily", "hourly"

# Una família: la matriu que pren, les seves tres funcions, i quins dels pronòstics que
# `predict` torna són què.
#
# `attribute` és `None` on no hi ha cap mètode i on no se'n vol cap. Les prediccions
# ingènues perquè la seva atribució seria la columna que copien. L'MLP perquè TreeSHAP
# no s'hi aplica i cap mètode exacte no és assumible dia a dia. I la xarxa de l'article
# perquè no és un model del qual s'esperi cap explicació: hi és per dir com de lluny
# són LEAR i l'arbre, que són els que s'interpreten, del que es pot arribar a fer amb
# aquestes dades.
#
# `members` són els models ajustats que entren a la combinació d'una execució, i
# `headlines` els que es comparen a les figures. Per a LEAR no són els mateixos: el que
# s'ajusta són les cinc finestres i el que es compara amb els altres models és la seva
# mitjana, que no és cap model. Les ingènues no ajusten res i no són membres de res.
#
# `distribution` és `None` a tot el que prediu 24 preus i prou. On no ho és, porta la
# distribució que la família treu i la funció que en dóna els paràmetres d'un dia: el
# camí endavant els desa al costat de les prediccions, perquè una distribució no són 24
# preus i tot el que llegeix una execució n'espera 24.
Family = collections.namedtuple(
    "Family", "matrix fit predict attribute distribution members headlines")


def matrices(data):
    """Les dues matrius per nom, que és com una família demana la seva."""
    return {DAILY: data.daily, HOURLY: data.hourly}


def families():
    """Cada família: quina matriu fa servir i, per pronòstic, les seves tres funcions.

    Cada model porta els seus números al seu propi
    `models/<nom>/<nom>_hyperparameters.py` i els pren per defecte, de manera que
    aquí no se'n lliga cap i el camí endavant tampoc no en sap res.
    """
    found = {
        "naive": Family(DAILY, naive.fit, naive.predict, None, None,
                        members=(), headlines=naive.NAMES),
        lear.NAME: Family(DAILY, lear.fit, lear.predict, lear.attribute, None,
                          members=lear.window_names(), headlines=(lear.NAME,)),
        boosting.NAME: Family(HOURLY, boosting.fit, boosting.predict,
                              boosting.attribute, None,
                              members=(boosting.NAME,), headlines=(boosting.NAME,)),
        mlp.NAME: Family(DAILY, mlp.fit, mlp.predict, None, None,
                         members=(mlp.NAME,), headlines=(mlp.NAME,)),
        dnn.NAME: Family(DAILY, dnn.fit, dnn.predict, None, None,
                         members=(dnn.NAME,), headlines=(dnn.NAME,)),
    }
    # Una família per distribució de sortida, que és el que fa l'article. Totes dues
    # són la mateixa xarxa amb un cap diferent, i el que les distingeix se'ls lliga a
    # `ddnn.bound`.
    for name, distribution in DISTRIBUTIONS.items():
        fit, predict, predicted = ddnn.bound(name, distribution)
        found[name] = Family(DAILY, fit, predict, None, predicted,
                             members=(name,), headlines=(name,))
    return found


def hyperparameters(family):
    """Els números del model d'una família, del seu fitxer i del que n'hagi cercat.

    Tot el que hi és en majúscules, que és la forma que tenen; així `run.json` els
    recull sense que ningú n'hagi de mantenir una llista.

    Una família que es cerca els seus números els té al disc i no al mòdul, i els
    demana aquí perquè aquest és l'únic lloc que respon "els números d'aquesta
    família". Sense això el `run.json` d'aquella família no portaria el que ha fet la
    predicció, que és justament el que l'ha de portar quan el commit ja no ho fixa.
    """
    try:
        module = importlib.import_module(f"epf.models.{family}.{family}_hyperparameters")
    except ModuleNotFoundError:
        return {}
    found = {name: getattr(module, name) for name in dir(module) if name.isupper()}
    if hasattr(module, "searched"):
        found |= module.searched()
    return found
