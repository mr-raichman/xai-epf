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
# per tant no les té en unitats comparables entre elles, ni tampoc entre hores.
# `values` és (membre, columna, hora) per a un dia i `base` és (membre, hora): el
# valor del qual parteix cada contribució, de manera que la base més la suma de les
# contribucions torna la predicció i això es pot comprovar. `forecasts` és
# (membre, hora) només per a un conjunt com LEAR, on cada finestra té la seva pròpia
# predicció i la que es desa al CSV és la mitjana; per a una família d'un sol model
# és None, perquè la predicció del membre ja és la del CSV.
Attribution = collections.namedtuple("Attribution",
                                     "members columns values base forecasts units")

# Membres d'una família que ajusta un sol model.
ONE = ("model",)

PRICE_UNITS = "EUR/MWh"

# Marge de la comprovació d'additivitat. Les contribucions es desen en `float32` i
# el residu mesurat és de l'ordre de 1e-5 EUR/MWh, de manera que això hi deixa dos
# ordres de magnitud i és molt per sota de qualsevol diferència que vulgui dir res
# en un preu.
TOLERANCE = 1e-3


def path(folder, forecast):
    return folder / f"attribution_{forecast}.npz"


def save(folder, forecast, days, attributions):
    """Desa l'atribució de tots els dies d'un pronòstic en un sol fitxer comprimit."""
    first = attributions[0]
    arrays = {
        "values": np.stack([one.values for one in attributions]).astype(np.float32),
        "base": np.stack([one.base for one in attributions]).astype(np.float32),
        "days": np.array([str(day.date()) for day in days]),
        "members": np.array(first.members),
        "columns": np.array(first.columns),
        "units": np.array(first.units)}
    if first.forecasts is not None:
        arrays["forecasts"] = np.stack(
            [one.forecasts for one in attributions]).astype(np.float32)
    np.savez_compressed(path(folder, forecast), **arrays)


def refuse_inconsistent(folder, forecast, predicted):
    """Falla si el que s'ha desat no reprodueix la predicció que hi ha al costat.

    És la comprovació que justifica desar la base. Per a una família que atribueix
    en EUR/MWh, la base més la suma de les contribucions ha de tornar la predicció;
    per a un conjunt que atribueix en un espai transformat, on aquella suma no és
    en EUR/MWh, qui l'ha de tornar és la mitjana de les prediccions dels membres.
    Llegeix el que hi ha al disc i no el que hi havia a la memòria, de manera que
    també agafa un error en desar. Torna el pitjor residu, o None si l'atribució no
    porta prou per comprovar res.
    """
    _, stored = load(folder, forecast)
    if stored.units == PRICE_UNITS:
        rebuilt = stored.base[:, 0] + stored.values[:, 0].sum(axis=1)
        whose = "la base més les contribucions"
    elif stored.forecasts is not None:
        rebuilt = stored.forecasts.mean(axis=1)
        whose = "la mitjana de les prediccions dels membres"
    else:
        return None
    worst = float(np.abs(rebuilt - predicted.to_numpy()).max())
    if worst > TOLERANCE:
        raise ValueError(f"l'atribució desada de {forecast} no reprodueix la seva "
                         f"predicció: {whose} se n'allunya {worst:.3g} EUR/MWh, per "
                         f"sobre del marge de {TOLERANCE}")
    return worst


def load(folder, forecast):
    """Els dies desats i l'atribució de tots ells, com una sola `Attribution`.

    Les execucions anteriors a la base no la porten, i aleshores `base` surt None en
    lloc de fer fallar la lectura.
    """
    stored = np.load(path(folder, forecast))
    return stored["days"], Attribution(
        members=list(stored["members"]),
        columns=list(stored["columns"]),
        values=stored["values"],
        base=stored["base"] if "base" in stored else None,
        forecasts=stored["forecasts"] if "forecasts" in stored else None,
        units=str(stored["units"]))
