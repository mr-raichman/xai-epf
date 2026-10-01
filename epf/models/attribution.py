"""Atribució d'una execució: quant va pesar cada columna en cada hora de lliurament.

Una execució no desa cap model ajustat, de manera que l'atribució de cada dia es
desa tal com surt del model que el va predir, amb signe i sense mitjanar res: així
es pot rellegir després per estació, per nivell de preu o per un sol dia sense
tornar a ajustar.
"""
import collections

import numpy as np

# `members` és un sol element per a les famílies d'un sol model, i les finestres de
# calibratge per a LEAR, que escala cada finestra amb la seva pròpia mediana i MAD i
# per tant no les té en unitats comparables entre elles. `values` és
# (membre, columna, hora) per a un dia.
Attribution = collections.namedtuple("Attribution", "members columns values units")

# Membres d'una família que ajusta un sol model.
ONE = ("model",)

PRICE_UNITS = "EUR/MWh"


def path(folder, forecast):
    return folder / f"attribution_{forecast}.npz"


def save(folder, forecast, days, attributions):
    """Desa l'atribució de tots els dies d'un pronòstic en un sol fitxer comprimit."""
    first = attributions[0]
    np.savez_compressed(
        path(folder, forecast),
        values=np.stack([one.values for one in attributions]).astype(np.float32),
        days=np.array([str(day.date()) for day in days]),
        members=np.array(first.members),
        columns=np.array(first.columns),
        units=np.array(first.units))


def load(folder, forecast):
    """(dies, membres, columnes, unitats, valors) tal com es van desar."""
    stored = np.load(path(folder, forecast))
    return (stored["days"], list(stored["members"]), list(stored["columns"]),
            str(stored["units"]), stored["values"])
