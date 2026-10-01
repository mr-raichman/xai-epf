"""Conjunt d'informació de l'etapa E1: el preu del mercat diari i res més.

Els 96 retards de preu i el dia de la setmana, que és el model de només preu de
Lago et al. (2021). Tot és calculable molt abans que tanqui la subhasta del dia
anterior al lliurament, perquè els preus del dia d-1 es van publicar el d-2.
"""
from epf.data import prices
from epf.features import matrix

STAGE = "e1"

# Retards que admet l'etapa. L'any més antic que pot entrenar és el primer del preu,
# el 2014, perquè no hi entra cap sèrie que comenci més tard.
LAGS = matrix.PRICE_LAGS


def load():
    """Les dues matrius de l'etapa: la diària i la per hora, ja alineades."""
    daily = matrix.build(prices.load(), LAGS)
    return daily, matrix.per_hour(*daily)
