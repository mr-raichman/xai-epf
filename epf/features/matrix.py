"""Construcció de la matriu de característiques a partir de les sèries carregades.

Una fila per dia de lliurament i 24 columnes de sortida, que és el marc
multivariant de la decisió 31: el model prediu les 24 hores alhora perquè el preu
d'una hora no és independent de la resta del seu dia.

Tota columna d'entrada ha de ser calculable, amb la mateixa definició, abans de
les 12:00 CET del dia anterior al lliurament (decisió 42). Els preus de fa un dia
compleixen: la subhasta del dia d-1 va tancar el dia d-2, així que ja eren
públics molt abans del tancament que fixa el preu del dia d.
"""
import pandas as pd

from epf.config.market import HOURS_PER_DAY
from epf.data.omie import HOUR_COLUMNS

# Els mateixos retards que fa servir LEAR, el model de referència de R1: ahir,
# abans d'ahir, fa tres dies i fa una setmana. Els tres primers porten la inèrcia
# recent i el setè el mateix dia de la setmana anterior.
PRICE_LAGS = (1, 2, 3, 7)

DAY_OF_WEEK = "day_of_week"


def build(prices, lags=PRICE_LAGS):
    """Torna (X, y) alineats per data de lliurament, sense cap fila incompleta.

    L'alineació és per data i no per posició. `shift(lag)` a seques desplaça per
    posició, així que amb un dia absent la fila següent agafaria com a "ahir" el
    dia anterior a l'absent sense dir-ho; `shift(lag, freq="D")` desplaça l'índex
    en dies i el `reindex` deixa buida la fila que no té el dia que li toca, de
    manera que desapareix en comptes d'aparellar-se amb el dia equivocat.
    """
    if list(prices.columns) != HOUR_COLUMNS:
        raise ValueError(f"s'esperaven les columnes {HOUR_COLUMNS}, i no {list(prices.columns)}")

    blocks = [prices.shift(lag, freq="D").reindex(prices.index).add_prefix(f"lag{lag}_")
              for lag in lags]
    blocks.append(prices.index.dayofweek.to_series(index=prices.index, name=DAY_OF_WEEK))
    features = pd.concat(blocks, axis="columns", sort=False)

    complete = features.notna().all(axis="columns")
    return features[complete], prices[complete]


def feature_names(lags=PRICE_LAGS):
    """Els noms de les columnes en l'ordre en què `build` les posa.

    Existeixen perquè SHAP imprimeix el que la matriu es diu a si mateixa, i una
    gràfica amb `feature_0` no diu res a ningú.
    """
    names = [f"lag{lag}_{hour}" for lag in lags for hour in HOUR_COLUMNS]
    return names + [DAY_OF_WEEK]


def horizon_names():
    """Els noms de les 24 sortides, una per hora del dia de lliurament."""
    return list(HOUR_COLUMNS[:HOURS_PER_DAY])
