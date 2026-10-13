"""La matriu diària com la llegeixen les dues xarxes: per blocs, i amb el dia de la
setmana en una sola columna.

Un bloc és una sèrie a un retard, les seves 24 hores, o el dia de la setmana: el que les
cerques de Lago et al. (2021) i de Marcjasz et al. (2023) engeguen i apaguen. Totes dues
donen el dia a la xarxa com un enter, de dilluns 0 a diumenge 6, i no en set indicadors
com fa la matriu per a la resta de famílies. Viu aquí, a sobre de les dues carpetes,
perquè les dues xarxes en llegeixen igual.
"""

from epf.matrix import DAY_OF_WEEK_COLUMNS

# El bloc del dia de la setmana, que la matriu escriu en set indicadors i les xarxes
# llegeixen en una sola columna, que porta aquest nom.
DAY = "day_of_week"


def blocks(columns):
    """Les columnes de la matriu per bloc: cada sèrie a cada retard, amb les seves 24
    hores, i el dia de la setmana.

    El bloc és el nom de la columna sense el que va després de l'últim guió baix, que és
    l'hora o el número del dia, de manera que surten de la matriu i no d'una llista que
    se'n pugui separar.
    """
    found = {}
    for column in columns:
        found.setdefault(column.rsplit("_", 1)[0], []).append(column)
    return found


def columns_of(columns, names):
    """Les columnes dels blocs anomenats, en l'ordre de la matriu."""
    found = blocks(columns)
    return [column for name in found if name in names for column in found[name]]


def inputs(X, names):
    """Les entrades dels blocs anomenats, amb el dia de la setmana al davant en una sola
    columna, `DAY`, de dilluns 0 a diumenge 6, com el codi dels dos articles el dóna a la
    xarxa.

    Els set indicadors de la matriu no hi entren: escalats per la mediana, cada un es
    dividiria per una MAD de zero.
    """
    kept = [column for column in columns_of(X.columns, names)
            if column not in DAY_OF_WEEK_COLUMNS]
    found = X[kept].astype(float)
    if DAY in names:
        found.insert(0, DAY, X[DAY_OF_WEEK_COLUMNS].to_numpy().argmax(axis=1).astype(float))
    return found
