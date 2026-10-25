"""El que comparteixen les explicacions de tots els models.

La comprovació que la base més les contribucions torna la predicció es fa dues vegades:
a la passada, abans de desar, contra el marge del seu espai, i sobre el fitxer desat,
amb el mateix marge més el que hi pot haver mogut l'arrodoniment a `float32`.
"""

import numpy as np

from epf import Refusal
from epf.contract.matrix import INFORMATION

# Les unitats d'una contribució: un preu, o un preu a l'espai de la transformació de
# Lago et al. (2021), on la base més les contribucions no són un preu. I les d'una
# columna que no en té cap, com el calendari.
PRICE_UNITS = "EUR/MWh"
TRANSFORMED_UNITS = "espai transformat"
UNITLESS = "sense unitats"

# Els grups d'una columna que no surt de cap sèrie: el calendari, i el comptador de dies
# de la sonda de datació.
CALENDAR = "calendar"
DATE = "date"

# Marge de la comprovació d'additivitat, en EUR/MWh: molt per sota de qualsevol
# diferència que vulgui dir res en un preu.
TOLERANCE = 1e-3

# Marge de l'additivitat a l'espai transformat, en les seves unitats. En EUR/MWh val
# aquest marge per l'escala del preu de l'hora, desenes d'euros, i per la derivada de
# `sinh`, d'1 a 1.5 a prop de zero: l'ordre del de dalt.
TRANSFORMED_TOLERANCE = 1e-5

# El que es pot moure un nombre, en relatiu, en desar-lo a `float32`, que és com es
# desen les taules de més d'una dimensió: la meitat de l'èpsilon del format.
ROUNDING = np.finfo(np.float32).eps / 2


def group(column):
    """Torna el grup d'una columna: la sèrie d'on surt, o el calendari.

    Una columna que només afegeix un model en dona el grup ell mateix.
    """
    for name in INFORMATION:
        if column == name or column.startswith(f"{name}_"):
            return name
    return CALENDAR


def check(fitted, rank):
    """Refusa un model ajustat si la seva explicació no torna la predicció.

    La base més les contribucions no se'n pot allunyar més que el marge del seu espai,
    `fitted.tolerance`. `rank` és el lloc de la configuració, que diu el missatge.
    """
    if fitted.gap > fitted.tolerance:
        raise Refusal(
            f"l'explicació de la configuració {rank} no torna la predicció: se "
            f"n'allunya {fitted.gap:.2g}, més que el marge de {fitted.tolerance:.2g}"
        )


def summed(base, values, axis):
    """Torna la base més les contribucions sumades sobre `axis`, en `float64`.

    Torna també la suma dels seus valors absoluts, que fita el que l'arrodoniment a
    `float32` hi pot haver mogut.
    """
    base = np.asarray(base, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return base + values.sum(axis=axis), np.abs(base) + np.abs(values).sum(axis=axis)


def explained_rows(found, per_day):
    """Torna les files de prova d'un fitxer, `found`, que són dels dies explicats.

    Són les primeres, `per_day` per dia, perquè la partició dona els dies de cada part
    en ordre de data i s'expliquen els primers dies de prova. Falla si els dies
    explicats del fitxer no són els primers dels seus dies de prova.
    """
    explained, test = found["days_explained"], found["days_test"]
    if not np.array_equal(explained, test[: len(explained)]):
        raise ValueError(
            "els dies explicats del fitxer no són els primers dels seus dies de prova"
        )
    return slice(0, per_day * len(explained))


def read_back(path, stored, tolerance):
    """Torna la distància i el marge de la suma que més se'n passa en un fitxer desat.

    El marge és `tolerance` més el que l'arrodoniment a `float32` pot haver mogut els
    termes de la suma, i una distància dins del marge vol dir que el fitxer torna el que
    explica. `stored` és la funció de la família que dona les sumes del fitxer.
    """
    worst = []
    with np.load(path) as found:
        for total, magnitude, target in stored(found):
            target = np.asarray(target, dtype=np.float64)
            distance = np.abs(total - target).ravel()
            margin = (tolerance + ROUNDING * (magnitude + np.abs(target))).ravel()
            # `argmax` i no `nanargmax`: una suma que no és un número ha de guanyar.
            index = int(np.argmax(distance - margin))
            worst.append((float(distance[index]), float(margin[index])))
    return max(worst, key=lambda pair: np.nan_to_num(pair[0] - pair[1], nan=np.inf))
