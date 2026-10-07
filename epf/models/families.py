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
from epf.models.lear import lear
from epf.models.mlp import mlp
from epf.models.naive import naive

DAILY, HOURLY = "daily", "hourly"

# Una família: la matriu que pren, les seves tres funcions, i quins dels pronòstics que
# `predict` torna són què.
#
# `attribute` és `None` on no hi ha cap mètode: les prediccions ingènues perquè la seva
# atribució seria la columna que copien, i l'MLP perquè TreeSHAP no s'hi aplica i cap
# mètode exacte no és assumible dia a dia; si mai és la família que s'interpreta, la
# importància per permutació, que no mira dins del model, hi arriba sense desar res a
# cada execució.
#
# `members` són els models ajustats que entren a la combinació d'una execució, i
# `headlines` els que es comparen a les figures. Per a LEAR no són els mateixos: el que
# s'ajusta són les cinc finestres i el que es compara amb els altres models és la seva
# mitjana, que no és cap model. Les ingènues no ajusten res i no són membres de res.
Family = collections.namedtuple("Family", "matrix fit predict attribute members headlines")


def matrices(data):
    """Les dues matrius per nom, que és com una família demana la seva."""
    return {DAILY: data.daily, HOURLY: data.hourly}


def families():
    """Cada família: quina matriu fa servir i, per pronòstic, les seves tres funcions.

    Cada model porta els seus números al seu propi
    `models/<nom>/<nom>_hyperparameters.py` i els pren per defecte, de manera que
    aquí no se'n lliga cap i el camí endavant tampoc no en sap res.
    """
    return {
        "naive": Family(DAILY, naive.fit, naive.predict, None,
                        members=(), headlines=naive.NAMES),
        lear.NAME: Family(DAILY, lear.fit, lear.predict, lear.attribute,
                          members=lear.window_names(), headlines=(lear.NAME,)),
        boosting.NAME: Family(HOURLY, boosting.fit, boosting.predict, boosting.attribute,
                              members=(boosting.NAME,), headlines=(boosting.NAME,)),
        mlp.NAME: Family(DAILY, mlp.fit, mlp.predict, None,
                         members=(mlp.NAME,), headlines=(mlp.NAME,)),
    }


def hyperparameters(family):
    """Els números del model d'una família, llegits del seu propi fitxer.

    Tot el que hi és en majúscules, que és la forma que tenen; així `run.json` els
    recull sense que ningú n'hagi de mantenir una llista.
    """
    try:
        module = importlib.import_module(f"epf.models.{family}.{family}_hyperparameters")
    except ModuleNotFoundError:
        return {}
    return {name: getattr(module, name) for name in dir(module) if name.isupper()}
