"""Preu espanyol del mercat diari, unint ESIOS i OMIE en una sola sèrie."""
import datetime

import pandas as pd

from epf.data import esios, omie

# Primer dia de l'arxiu públic d'OMIE. Abans, el preu és el d'ESIOS, idèntic al
# d'OMIE a tots els dies en què coincideixen.
OMIE_FROM = datetime.date(2018, 1, 1)

SPAIN = "España"


def load():
    """Preus en EUR/MWh, un dia de lliurament per fila i una hora per columna."""
    early = esios.load_price(SPAIN)
    early = early[early.index < pd.Timestamp(OMIE_FROM)]
    return pd.concat([early, omie.load()])
