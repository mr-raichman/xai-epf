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

from epf.models.boosting import boosting
from epf.models.boosting import boosting_hyperparameters
from epf.models.ddnn import ddnn, ddnn_hyperparameters
from epf.models.ddnn.distributions import FAMILIES as DISTRIBUTIONS
from epf.models.dnn import dnn, dnn_hyperparameters
from epf.models.lear import lear, lear_hyperparameters
from epf.models.naive import naive

DAILY, HOURLY = "daily", "hourly"

# Una família: la matriu que pren, les seves tres funcions, i quins dels pronòstics que
# `predict` torna són què.
#
# `attribute` és `None` on no hi ha cap mètode i on no se'n vol cap. Les prediccions
# ingènues perquè la seva atribució seria la columna que copien. LEAR i la xarxa de
# l'article perquè són referències i no models dels quals s'esperi cap explicació: LEAR
# és el que els altres han de superar, i la xarxa diu com de lluny queden els models
# que s'expliquen del que es pot arribar a fer amb aquestes dades.
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
#
# `hyperparameters` és el mòdul que porta els números del model, o `None` on no en té
# cap. Hi és en lloc de derivar-lo del nom de la família: dues famílies poden sortir
# d'una sola carpeta, cosa que fan les dues distribucionals, i mentre el nom es
# derivava, el `run.json` d'aquelles dues no portava cap número i no es queixava.
Family = collections.namedtuple(
    "Family",
    "matrix fit predict attribute distribution hyperparameters members headlines")


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
        "naive": Family(DAILY, naive.fit, naive.predict, None, None, None,
                        members=(), headlines=naive.NAMES),
        lear.NAME: Family(DAILY, lear.fit, lear.predict, None, None,
                          lear_hyperparameters,
                          members=lear.window_names(), headlines=(lear.NAME,)),
        boosting.NAME: Family(HOURLY, boosting.fit, boosting.predict,
                              boosting.attribute, None, boosting_hyperparameters,
                              members=(boosting.NAME,), headlines=(boosting.NAME,)),
        dnn.NAME: Family(DAILY, dnn.fit, dnn.predict, None, None,
                         dnn_hyperparameters,
                         members=(dnn.NAME,), headlines=(dnn.NAME,)),
    }
    # Una família per distribució de sortida, que és el que fa l'article. Totes dues
    # són la mateixa xarxa amb un cap diferent, i el que les distingeix se'ls lliga a
    # `ddnn.bound`.
    for name, distribution in DISTRIBUTIONS.items():
        fit, predict, predicted = ddnn.bound(name, distribution)
        found[name] = Family(DAILY, fit, predict, None, predicted,
                             ddnn_hyperparameters,
                             members=(name,), headlines=(name,))
    return found


def hyperparameters(family):
    """Els números del model d'una família, del seu mòdul i del que n'hagi cercat.

    Tot el que el mòdul té en majúscules, que és la forma que tenen; així `run.json`
    els recull sense que ningú n'hagi de mantenir una llista. Quin mòdul és ho diu el
    registre i no el nom de la família: dues famílies poden compartir carpeta.

    Una família que es cerca els seus números els té al disc i no al mòdul, i els
    demana aquí perquè aquest és l'únic lloc que respon "els números d'aquesta
    família". Sense això el `run.json` d'aquella família no portaria el que ha fet la
    predicció, que és justament el que l'ha de portar quan el commit ja no ho fixa.
    """
    module = families()[family].hyperparameters
    if module is None:
        return {}
    found = {name: getattr(module, name) for name in dir(module) if name.isupper()}
    if hasattr(module, "searched"):
        found |= module.searched()
    return found
