"""El que comparteixen les explicacions de tots els models: el grup de cada columna, les
unitats d'una contribució i la comprovació que la base més les contribucions torna la
predicció que expliquen.

La comprovació es fa dues vegades: a la memòria, abans de desar, contra el marge del seu
espai, i sobre el fitxer desat, amb el mateix marge més el que hi pot haver mogut
l'arrodoniment a `float32`. Quines sumes es tornen a fer sobre el fitxer ho diu cada
família.
"""
import numpy as np

from epf.contract.matrix import INFORMATION

# Les unitats d'una contribució: un preu, un preu a l'espai de la transformació de Lago
# et al. (2021), on la base més les contribucions no són un preu, o cap, que són
# l'asimetria i el pes de la cua d'una distribució.
PRICE_UNITS = "EUR/MWh"
TRANSFORMED_UNITS = "espai transformat"
UNITLESS = "sense unitats"

# Marge de la comprovació d'additivitat, en EUR/MWh: molt per sota de qualsevol
# diferència que vulgui dir res en un preu. Es comprova sobre el que hi ha a la memòria,
# abans de desar, i el residu de cada llavor s'imprimeix al costat de les seves
# puntuacions.
TOLERANCE = 1e-3

# Marge de l'additivitat a l'espai transformat, a part perquè és en les seves unitats i
# no en un preu: en EUR/MWh val tant com el de dalt, perquè a prop d'1 la derivada de
# `sinh` és d'1,5 i la MAD de les files d'ajust va en desenes d'euros.
TRANSFORMED_TOLERANCE = 1e-4

# El que es pot moure un nombre, en relatiu, en desar-lo a `float32`, que és com es desen
# les taules de més d'una dimensió: la meitat de l'èpsilon del format.
ROUNDING = np.finfo(np.float32).eps / 2


def group(column):
    """El grup d'una columna a la importància per grups: la sèrie del conjunt d'informació
    d'on surt o, la resta, el calendari. Una columna que només afegeix un model en dona el
    grup ell mateix."""
    for name in INFORMATION:
        if column == name or column.startswith(f"{name}_"):
            return name
    return "calendar"


def check(fitted, family, seed, rank):
    """Falla si la base més les contribucions d'un model ajustat se separen de la predicció
    que expliquen més que el marge del seu espai."""
    if fitted.gap > fitted.tolerance:
        raise ValueError(f"l'explicació de {family}, llavor {seed}, configuració {rank}, no "
                         f"torna la seva predicció: se n'allunya {fitted.gap:.3g}, per sobre "
                         f"del marge de {fitted.tolerance}")


def summed(base, values, axis):
    """La base més les contribucions sumades sobre `axis`, en `float64`, i la suma dels
    seus valors absoluts, que fita el que l'arrodoniment a `float32` hi pot haver mogut."""
    base = np.asarray(base, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return base + values.sum(axis=axis), np.abs(base) + np.abs(values).sum(axis=axis)


def explained_rows(found, per_day):
    """Les files de les prediccions de prova d'un fitxer que són dels dies explicats.

    Són les primeres, perquè la partició dona els dies de cada part en ordre de data i
    s'expliquen els primers dies de prova; el fitxer ho ha de confirmar, amb els seus dies
    explicats al començament dels seus dies de prova.
    """
    explained, test = found["days_explained"], found["days_test"]
    if not np.array_equal(explained, test[:len(explained)]):
        raise ValueError("els dies explicats del fitxer no són els primers dels seus dies "
                         "de prova")
    return slice(0, per_day * len(explained))


def read_back(path, stored, tolerance):
    """Quant es passen del seu marge les sumes d'un fitxer desat, en el pitjor cas: el
    marge és el de la comprovació a la memòria més el que l'arrodoniment a `float32`
    pot haver mogut els termes de cada suma, i zero o menys vol dir que el fitxer torna
    el que explica."""
    worst = []
    with np.load(path) as found:
        for total, magnitude, target in stored(found):
            target = np.asarray(target, dtype=np.float64)
            margin = tolerance + ROUNDING * (magnitude + np.abs(target))
            worst.append(np.max(np.abs(total - target) - margin))
    return float(np.max(worst))
