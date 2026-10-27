"""Els information sets: les sèries que llegeixen els models, i a quins retards.

És l'únic que s'edita per canviar una entrada. Cada sèrie es declara una vegada, a
`SERIES`, i el seu tipus decideix, sense cap altra declaració, com es baixa, com
s'omple i com la llegeixen els models i les figures. Cada information set s'escriu
sencer, i cap no és especial. El calendari hi entra sempre, i no és de cap set.
"""

import collections

from epf import Refusal

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

# El que es prediu: el número d'ESIOS del preu; el nom de la seva sèrie; com es diu a
# les figures dels models i el títol de la seva figura del dataset; i la sèrie que pesa
# cada hora a la referència.
Target = collections.namedtuple("Target", "id name label title weights")
TARGET = Target(
    600,
    "price",
    "preu del dia abans",
    "Preu horari del mercat diari espanyol",
    "demand_fcst",
)
PRICE = TARGET.name

# El dia que llegeix la referència, en dies abans del lliurament: la mitjana del preu
# del dia abans, cada hora pesada per la previsió de demanda del mateix dia.
REFERENCE_LAG = 1

# Una sèrie: els indicadors d'ESIOS que suma, en l'ordre en què se sumen; el seu tipus;
# com es diu a les figures dels models, i el títol de la seva figura del dataset.
Series = collections.namedtuple("Series", "ids kind label title")

# Cada sèrie, una sola vegada, en l'ordre de les columnes del dataset. La solar és
# només la fotovoltaica. L'hidràulica és la de les unitats de gestió hidràulica (UGH) i
# la de fora d'elles, i el bombeig, el seu turbinatge.
SERIES = {
    PRICE: Series([TARGET.id], "price", TARGET.label, TARGET.title),
    "demand_fcst": Series(
        [1775], "forecast", "demanda prevista", "Previsió D+1 de la demanda"
    ),
    "wind_fcst": Series(
        [1777], "forecast", "eòlica prevista", "Previsió D+1 de l'eòlica"
    ),
    "solar_fcst": Series(
        [1779],
        "forecast",
        "fotovoltaica prevista",
        "Previsió D+1 de la fotovoltaica",
    ),
    "nuclear_sched": Series(
        [4], "schedule", "nuclear programada", "Generació nuclear programada (PBF)"
    ),
    "coal_gas_sched": Series(
        [7, 8, 9, 11, 17],
        "schedule",
        "carbó i gas programats",
        "Generació de carbó i gas programada (PBF)",
    ),
    "hydro_sched": Series(
        [1, 2, 3],
        "schedule",
        "hidràulica programada",
        "Generació hidràulica programada, amb el turbinatge de bombeig (PBF)",
    ),
}

# Un information set: les sèries de `SERIES` que llegeix, en l'ordre de les seves
# columnes a les matrius, cada una amb els dies abans del lliurament en què hi entra; i
# si les famílies explicades hi prediuen la distància de cada hora a la referència. Les
# previsions D+1 són del mateix dia que es lliura, i el programa, del dia abans, tancat
# dos dies abans i a l'abast de tothom abans del tancament de la subhasta.
InformationSet = collections.namedtuple("InformationSet", "series reference")

# Cada information set, pel nom que pren `--set`.
WITH_PRICE = InformationSet(
    {
        PRICE: [1],
        "demand_fcst": [0],
        "wind_fcst": [0],
        "solar_fcst": [0],
        "nuclear_sched": [1],
        "coal_gas_sched": [1],
        "hydro_sched": [1],
    },
    reference=False,
)
WITHOUT_PRICE = InformationSet(
    {
        "demand_fcst": [0],
        "wind_fcst": [0],
        "solar_fcst": [0],
        "nuclear_sched": [1],
        "coal_gas_sched": [1],
        "hydro_sched": [1],
    },
    reference=True,
)

# Un information set resolt: el seu nom; les seves sèries amb els retards, com les
# declara; la sèrie que pesa la referència, o `None` si no en té; i els dies que surten
# de l'entrenament a cada costat d'una setmana retinguda, el retard més llarg al qual
# hi entra el preu, com a entrada o per la referència, perquè una setmana retinguda i
# els dies que en llegeixen el preu es miren l'un a l'altre.
Resolved = collections.namedtuple("Resolved", "name series reference gap")


def column(name, lag):
    """Torna el nom d'una sèrie a un retard, `price_lag1`, o tal qual al retard 0."""
    return f"{name}_lag{lag}" if lag else name


def _resolved(name, declared):
    """Torna l'information set `declared`, de nom `name`, resolt, sense comprovar-lo."""
    reference = TARGET.weights if declared.reference else None
    lags = declared.series.get(PRICE, ())
    gap = max((*lags, *((REFERENCE_LAG,) if reference else ())), default=0)
    return Resolved(name, declared.series, reference, gap)


def _refused(found):
    """Refusa `SERIES` o uns information sets resolts, `found`, inservibles.

    Diu el primer motiu que troba: de `SERIES` abans que de cap set, i de cada set en
    el seu ordre.
    """
    for name, row in SERIES.items():
        if row.kind not in KINDS:
            raise Refusal(
                f'la sèrie {name} té un tipus que no existeix, "{row.kind}"; els '
                f"tipus: {', '.join(KINDS)}"
            )
    for one in found.values():
        ids, columns = {}, set()
        for name, lags in one.series.items():
            if name not in SERIES:
                raise Refusal(f"{one.name} llegeix la sèrie {name}, que SERIES no té")
            for number in SERIES[name].ids:
                if number in ids:
                    raise Refusal(
                        f"l'indicador {number} és a dues sèries de {one.name}, "
                        f"{ids[number]} i {name}"
                    )
                ids[number] = name
            for lag in lags:
                if lag < 0:
                    raise Refusal(
                        f"la sèrie {name} de {one.name} té un retard negatiu, {lag}"
                    )
                if name == PRICE and lag == 0:
                    raise Refusal(
                        f"el preu entra a {one.name} al retard 0, el dia que es prediu"
                    )
                built = column(name, lag)
                if built in columns:
                    raise Refusal(f"la columna {built} es fa dues vegades a {one.name}")
                columns.add(built)
    if any(one.reference for one in found.values()) and TARGET.weights not in SERIES:
        raise Refusal(f"la referència es pesa amb {TARGET.weights}, que SERIES no té")


def every():
    """Torna cada information set resolt, pel seu nom, si tots es poden fer servir.

    Els sets són els `InformationSet` d'aquest mòdul, en el seu ordre.
    """
    found = {
        name: _resolved(name, value)
        for name, value in globals().items()
        if isinstance(value, InformationSet)
    }
    _refused(found)
    return found


def resolved(name):
    """Torna l'information set `name` resolt, i el refusa com `every`.

    Refusa també un nom que no és de cap information set.
    """
    found = every()
    if name not in found:
        raise Refusal(
            f'"{name}" no és cap information set; els information sets: '
            f"{', '.join(found)}"
        )
    return found[name]


def in_dataset():
    """Torna cada sèrie del dataset, en l'ordre de `SERIES`.

    Són el preu, cada sèrie que llegeix algun information set i la que pesa la
    referència, si algun en té.
    """
    sets = every().values()
    read = {PRICE} | {name for one in sets for name in one.series}
    read |= {one.reference for one in sets if one.reference}
    return {name: row for name, row in SERIES.items() if name in read}


def recorded(information_set):
    """Torna un information set resolt com el desen els registres.

    El seu nom; de cada sèrie, els indicadors, el tipus i els retards, sense els noms de
    figura; el que es prediu; i la sèrie que pesa la referència, o `None`.
    """
    return {
        "name": information_set.name,
        "series": {
            name: {"ids": SERIES[name].ids, "kind": SERIES[name].kind, "lags": lags}
            for name, lags in information_set.series.items()
        },
        "target": {"id": TARGET.id, "name": TARGET.name},
        "reference": information_set.reference,
    }
