"""Atribució d'una execució: quant va pesar cada columna en cada hora de lliurament.

Una execució no desa cap model ajustat, de manera que l'atribució de cada dia es
desa tal com surt del model que el va predir, amb signe i sense mitjanar res: així
es pot rellegir després per estació, per nivell de preu o per un sol dia sense
tornar a ajustar.
"""
import collections

import numpy as np

# `members` és un sol element per a les famílies d'un sol model, i un per model per a
# una família de diversos, on cada un escala les seves pròpies files i per tant no
# tenen unitats comparables entre ells, ni tampoc entre hores.
# `values` és (membre, columna, hora) per a un dia i `base` és (membre, hora): el
# valor del qual parteix cada contribució, de manera que la base més la suma de les
# contribucions torna la predicció i això es pot comprovar. `forecasts` és
# (membre, hora), la predicció de cada membre en EUR/MWh, allà on l'atribució no és en
# EUR/MWh: és el que es compara amb el CSV, la d'un sol membre o la mitjana de diversos,
# i sense ella aquella comprovació no té res a comparar. On l'atribució és en EUR/MWh
# és None, perquè la base més les contribucions ja és el preu. `transformed` és la
# predicció de cada membre a les unitats de l'atribució i no en EUR/MWh, i hi és
# només on aquelles dues no coincideixen: és contra ella que es comprova
# l'additivitat quan la suma de les contribucions no és un preu.
Attribution = collections.namedtuple(
    "Attribution", "members columns values base forecasts transformed units")

# Membres d'una família que ajusta un sol model.
ONE = ("model",)

PRICE_UNITS = "EUR/MWh"

# Marge de la comprovació d'additivitat. Les contribucions es desen en `float32` i
# el residu mesurat és de l'ordre de 1e-5 EUR/MWh, de manera que això hi deixa dos
# ordres de magnitud i és molt per sota de qualsevol diferència que vulgui dir res
# en un preu.
TOLERANCE = 1e-3

# Marge de l'additivitat quan les contribucions no són en EUR/MWh. Hi va a part
# perquè és a les unitats de l'atribució i no en un preu. Mesurat amb les cinc
# finestres de LEAR sobre valors transformats d'ordre 1: la identitat es compleix a
# 1,3e-15 en memòria, que és precisió doble, i després del pas a `float32` amb què es
# desa queda a 4,8e-7 sobre vuit dies de juny de 2023 i a 1,9e-6 sobre els 30 dies del
# període de prova. Això deixa un factor 50 de marge, i en EUR/MWh val tant com el
# marge de dalt: a prop d'1 la derivada de `sinh` és d'1,5 i la MAD d'una finestra va
# en desenes d'euros.
TRANSFORMED_TOLERANCE = 1e-4


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
    for name in ("forecasts", "transformed"):
        if getattr(first, name) is not None:
            arrays[name] = np.stack(
                [getattr(one, name) for one in attributions]).astype(np.float32)
    np.savez_compressed(path(folder, forecast), **arrays)


def refuse_nonadditive(forecast, stored):
    """Falla si la base i les contribucions d'un membre no tornen la seva predicció.

    És l'additivitat que justifica desar la base, i la comprovació és exacta per
    construcció: per a un model lineal la contribució d'una columna és el coeficient
    per la distància al fons i la base és la predicció al fons, de manera que la suma
    és `intercept_ + coef_ @ x`, que és la predicció.

    Quan les contribucions no són en EUR/MWh aquesta és l'única manera de
    comprovar-ho, perquè la suma és a l'espai transformat i `sinh` no deixa
    convertir-la: es compara amb la predicció del model al mateix espai. Sense això, la
    família amb l'atribució exacta per construcció seria l'única que no comprova res
    (`evidence.md` 5.4).
    """
    if stored.transformed is None:
        return
    worst = float(np.abs(stored.base + stored.values.sum(axis=2)
                         - stored.transformed).max())
    if worst > TRANSFORMED_TOLERANCE:
        raise ValueError(f"l'atribució desada de {forecast} no és additiva: la base "
                         f"més les contribucions s'allunyen {worst:.3g} de la "
                         f"predicció del membre, per sobre del marge de "
                         f"{TRANSFORMED_TOLERANCE}")


def refuse_inconsistent(folder, forecast, predicted):
    """Falla si el que s'ha desat no reprodueix la predicció que hi ha al costat.

    Dues comprovacions i no una. L'additivitat de cada membre és de
    `refuse_nonadditive`, i aquí es comprova que el que s'ha desat torni el CSV: per a
    una família que atribueix en EUR/MWh les dues coses són la mateixa, i per a una que
    atribueix a l'espai transformat qui ha de tornar el CSV és la mitjana de les
    prediccions desades dels seus membres, un o diversos.

    Llegeix el que hi ha al disc i no el que hi havia a la memòria, de manera que
    també agafa un error en desar. Torna el pitjor residu en EUR/MWh, o None si
    l'atribució no porta prou per comprovar res.
    """
    _, stored = load(folder, forecast)
    refuse_nonadditive(forecast, stored)
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
        transformed=stored["transformed"] if "transformed" in stored else None,
        units=str(stored["units"]))
