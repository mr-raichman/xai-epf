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

# Les tres funcions d'un pronòstic, cada una amb una feina. `attribute` és `None` on
# no hi ha cap mètode: les prediccions ingènues perquè la seva atribució seria la
# columna que copien, i l'MLP perquè TreeSHAP no s'hi aplica i cap mètode exacte és
# assumible dia a dia; si mai és la família que s'interpreta, la importància per
# permutació, que no mira dins del model, hi arriba sense desar res a cada execució.
Forecast = collections.namedtuple("Forecast", "fit predict attribute")

# `fits` diu si la família ajusta un model. Les ingènues no: copien una columna
# que ja es coneixia, i per això no entren a la combinació. No se'n deriva de `fit`,
# que hi és a totes, ni d'`attribute`, que l'MLP no té tot i ajustar.
Family = collections.namedtuple("Family", "matrix forecasts fits")


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
        "naive": Family(DAILY, {
            "previous_day": Forecast(naive.fit, naive.previous_day, None),
            "previous_week": Forecast(naive.fit, naive.previous_week, None),
            "similar_day": Forecast(naive.fit, naive.similar_day, None),
        }, fits=False),
        "lear": Family(DAILY, {
            "lear": Forecast(lear.fit, lear.predict, lear.attribute)}, fits=True),
        "lightgbm": Family(HOURLY, {
            "lightgbm": Forecast(boosting.fit, boosting.predict, boosting.attribute)},
            fits=True),
        "mlp": Family(DAILY, {"mlp": Forecast(mlp.fit, mlp.predict, None)}, fits=True),
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
