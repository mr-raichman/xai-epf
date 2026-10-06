"""El conjunt de dades del preu: el preu espanyol del mercat diari, de dues fonts.

ESIOS el serveix des del 2014 i OMIE publica el seu arxiu des del 2018, idèntic al
d'ESIOS a tots els dies en què coincideixen, i el conjunt és l'un abans del tall i
l'altre a partir. Les dues fonts viuen planes a `paths.PRICE` perquè són un sol
conjunt de dades, i els seus noms de fitxer no xoquen.

Executar el mòdul fa existir el conjunt: el baixa sencer i diu què queda a disc.
Comprovar-lo és d'una altra banda, `handle_data.py`, que és l'única que comprova.
"""
import datetime

import pandas as pd

from epf.data import esios, omie
from epf.paths import PRICE

# Indicador del preu a ESIOS. No és a `esios.SERIES` i no es llegeix amb
# `esios.load`: és l'única sèrie que porta diverses geografies que es trien en lloc
# de sumar-se, i d'aquestes la que es demana és Espanya.
ESIOS_PRICE = 600
SPAIN = "España"
PRICE_COUNTRIES = ("España", "Portugal", "Francia")

# Primer dia que ESIOS serveix el preu, mesurat: una sol·licitud directa a l'API no
# en torna cap abans. És un fet de la font i no una tria nostra, que viu a
# `parameters.py`. Alguna altra sèrie d'ESIOS comença unes hores abans.
ESIOS_FIRST_DAY = datetime.date(2014, 1, 1)

# Primer dia de l'arxiu públic d'OMIE.
OMIE_FROM = datetime.date(2018, 1, 1)


def from_esios(country, directory=PRICE):
    """Preu del mercat diari d'un país en EUR/MWh, i les hores publicades cada dia."""
    if country not in PRICE_COUNTRIES:
        raise KeyError(f"cap preu desat per a {country}; n'hi ha per a {PRICE_COUNTRIES}")
    stored = esios.values(directory)
    return esios.hourly([value for value in stored if value["geo_name"] == country])


def load(directory=PRICE):
    """Preus en EUR/MWh, un dia de lliurament per fila i una hora per columna."""
    early, _ = from_esios(SPAIN, directory)
    early = early[early.index < pd.Timestamp(OMIE_FROM)]
    return pd.concat([early, omie.load(directory)])


def download(directory=PRICE):
    """Baixa les dues fonts al directori del conjunt."""
    print("OMIE", flush=True)
    omie.download(directory)
    print("ESIOS", flush=True)
    esios.download(ESIOS_PRICE, directory)


def main():
    download()
    frame = load()
    print(f"{len(frame)} dies de preu, de {frame.index[0].date()} "
          f"a {frame.index[-1].date()}")
