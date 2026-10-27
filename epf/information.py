"""L'information set: les sèries que llegeixen els models, i a quins retards.

És l'únic que s'edita per canviar una entrada. Cada sèrie és una columna del dataset, i
el seu tipus decideix, sense cap altra declaració, com es baixa, com s'omple i com la
llegeixen els models i les figures. El calendari hi entra sempre, i no és de la taula.
"""

import collections

# El nom de la sèrie que es prediu.
PRICE = "price"

# El que decideix el tipus d'una sèrie: com n'agrega ESIOS cada hora, `average` o `sum`;
# la geografia que se'n pren i, amb `only`, si ha de ser l'única del fitxer o, sense, si
# es tria entre les que porta; amb `anchored`, si una hora que ESIOS no publica hi val
# zero on l'àncora la publica, i les de l'apagada en són l'única llacuna declarada; amb
# `at_the_hour`, si l'arbre i l'elastic net en llegeixen el valor de l'hora de la fila,
# i si no les 24 hores; i la unitat dels seus valors.
Kind = collections.namedtuple(
    "Kind", "aggregation geography only anchored at_the_hour unit"
)
KINDS = {
    # El preu del mercat diari, que es tria entre les sis geografies del seu fitxer.
    "price": Kind("average", "España", False, False, False, "EUR/MWh"),
    # Les previsions D+1 de Red Eléctrica.
    "forecast": Kind("average", "Península", True, False, True, "MW"),
    # El programa base de funcionament (PBF), en MWh per hora: des del 2025-10-01 ESIOS
    # el serveix com a energia per quart d'hora, i la mitjana en seria la quarta part.
    "schedule": Kind("sum", "Península", True, True, True, "MWh"),
}

# Una sèrie: els indicadors d'ESIOS que suma, en l'ordre en què se sumen; el seu tipus;
# els dies abans del lliurament en què entra a les matrius; com es diu a les figures
# dels models, i el títol de la seva figura del dataset.
Series = collections.namedtuple("Series", "ids kind lags label title")

# Cada sèrie, en l'ordre de les seves columnes a les matrius. Les previsions D+1 són del
# mateix dia que es lliura, i el programa, del dia abans, tancat dos dies abans i a
# l'abast de tothom abans del tancament de la subhasta. La solar és només la
# fotovoltaica. L'hidràulica és la de les unitats de gestió hidràulica (UGH) i la de
# fora d'elles, i el bombeig, el seu turbinatge.
INFORMATION = {
    PRICE: Series(
        (600,),
        "price",
        (1,),
        "preu del dia abans",
        "Preu horari del mercat diari espanyol",
    ),
    "demand_fcst": Series(
        (1775,), "forecast", (0,), "demanda prevista", "Previsió D+1 de la demanda"
    ),
    "wind_fcst": Series(
        (1777,), "forecast", (0,), "eòlica prevista", "Previsió D+1 de l'eòlica"
    ),
    "solar_fcst": Series(
        (1779,),
        "forecast",
        (0,),
        "fotovoltaica prevista",
        "Previsió D+1 de la fotovoltaica",
    ),
    "nuclear_sched": Series(
        (4,),
        "schedule",
        (1,),
        "nuclear programada",
        "Generació nuclear programada (PBF)",
    ),
    "coal_gas_sched": Series(
        (7, 8, 9, 11, 17),
        "schedule",
        (1,),
        "carbó i gas programats",
        "Generació de carbó i gas programada (PBF)",
    ),
    "hydro_sched": Series(
        (1, 2, 3),
        "schedule",
        (1,),
        "hidràulica programada",
        "Generació hidràulica programada, amb el turbinatge de bombeig (PBF)",
    ),
}

# Dies que surten de l'entrenament a cada costat d'una setmana retinguda: el retard més
# llarg del preu, perquè una setmana retinguda i els dies que en llegeixen el preu es
# miren l'un a l'altre.
GAP = max(INFORMATION[PRICE].lags)


def recorded():
    """Torna l'information set com el desen els registres, sense els noms de figura.

    De cada sèrie, els indicadors, el tipus i els retards.
    """
    return {
        name: {"ids": series.ids, "kind": series.kind, "lags": series.lags}
        for name, series in INFORMATION.items()
    }
