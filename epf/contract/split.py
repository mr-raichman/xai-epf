"""La partició de Trebbien et al. (2023), per setmanes barrejades amb la seed.

Les setmanes del període es reparteixen en entrenament, validació i prova. A diferència
del seu article, el preu del dia abans és una entrada, i per això l'entrenament perd el
dia de cada costat d'una setmana retinguda; i una setmana amb un dia que alguna entrada
no pot omplir no es reparteix.
"""

import collections

import numpy as np
import pandas as pd

from epf import Refusal
from epf.information import GAP
from epf.parameters import DAYS_PER_WEEK, SHARES, seeded

# Els dies de cada part d'una partició, en l'ordre en què es reparteixen les setmanes.
Split = collections.namedtuple("Split", "training validation test")

# Parts en què es parteix la validació perquè l'entrenament s'aturi quan qualsevol
# d'elles deixa de millorar, les de Trebbien et al. (2023).
VALIDATION_PARTS = 4

# Per què surt del període un dia al qual no falta res: la seva setmana en surt, o és
# dels últims, que no fan cap setmana sencera.
WEEK = "week"
NO_WHOLE_WEEK = "no_whole_week"


def deleted(period, lacking):
    """Torna els dies del període que no es reparteixen, cada un amb el seu motiu.

    Les setmanes es compten des del primer dia del període, i en surt tota la que té
    algun dia de `lacking`, les columnes que li falten per dia; i també els últims dies,
    que no fan cap setmana sencera. El motiu d'un dia de `lacking` són les seves
    columnes, i el de cada altre, `WEEK` o `NO_WHOLE_WEEK`. Els dies són
    `datetime.date`, en ordre.
    """
    days = pd.date_range(*period, freq="D")
    whole = len(days) - len(days) % DAYS_PER_WEEK
    found = {}
    for start in range(0, whole, DAYS_PER_WEEK):
        week = days[start : start + DAYS_PER_WEEK]
        if any(day in lacking for day in week):
            found |= {day.date(): lacking.get(day, WEEK) for day in week}
    found |= {day.date(): NO_WHOLE_WEEK for day in days[whole:]}
    return found


def _deal(seed, period, deleted, shares):
    """Torna els dies que es reparteixen i les setmanes de cada part, en ordre de seed.

    El període es talla en setmanes de set dies des del primer, i en surten els últims
    dies, que no en fan cap de sencera, i tota setmana amb algun dia de `deleted`. La
    resta es barregen amb la seed i es reparteixen en l'ordre de `Split`: a la
    validació, la seva part de `shares` arrodonida a un múltiple de `VALIDATION_PARTS`,
    perquè es parteixi en parts iguals; a la prova, la seva arrodonida; i a
    l'entrenament, la resta.
    """
    days = pd.date_range(*period, freq="D")
    weeks = len(days) // DAYS_PER_WEEK
    days = days[: weeks * DAYS_PER_WEEK]
    gone = days.isin(pd.to_datetime(list(deleted))).reshape(weeks, DAYS_PER_WEEK)
    kept = np.flatnonzero(~gone.any(axis=1))
    order = np.random.default_rng(seeded("split", seed)).permutation(kept)
    _, validation_share, test_share = shares
    validation = VALIDATION_PARTS * round(
        len(kept) * validation_share / 100 / VALIDATION_PARTS
    )
    test = round(len(kept) * test_share / 100)
    ends = np.cumsum([len(kept) - validation - test, validation, test])
    return days, [
        order[start:end] for start, end in zip([0, *ends[:-1]], ends, strict=True)
    ]


def _days_of(days, weeks):
    """Torna els dies de les setmanes `weeks`, en ordre de data."""
    return days[
        np.sort(
            np.concatenate(
                [
                    np.arange(week * DAYS_PER_WEEK, (week + 1) * DAYS_PER_WEEK)
                    for week in weeks
                ]
            )
        )
    ]


def split(seed, period, deleted, shares=SHARES, gap=GAP):
    """Torna els dies d'entrenament, de validació i de prova d'una seed.

    Surt de l'entrenament tot dia que tingui un dia retingut a `gap` dies o menys, a
    qualsevol costat, i no és de cap part. Una setmana amb algun dia de `deleted` no és
    retinguda: els dies del seu costat es reparteixen com qualsevol altre.
    """
    days, dealt = _deal(seed, period, deleted, shares)
    part_of_week = np.full(len(days) // DAYS_PER_WEEK, -1)
    for part, weeks in enumerate(dealt):
        part_of_week[weeks] = part
    part_of_day = np.repeat(part_of_week, DAYS_PER_WEEK)

    training = part_of_day == 0
    held_out = (part_of_day == 1) | (part_of_day == 2)
    beside = np.zeros(len(days), dtype=bool)
    for offset in range(1, gap + 1):
        beside[offset:] |= held_out[:-offset]
        beside[:-offset] |= held_out[offset:]
    return Split(
        days[training & ~beside], days[part_of_day == 1], days[part_of_day == 2]
    )


def validation_parts(seed, period, deleted, count=VALIDATION_PARTS, shares=SHARES):
    """Torna els dies de validació d'una seed en `count` parts iguals de setmanes.

    Les parts segueixen l'ordre en què la seed ha repartit les setmanes, de manera que
    cada una és de setmanes a l'atzar. Refusa unes setmanes de validació que no fan
    `count` parts iguals.
    """
    days, dealt = _deal(seed, period, deleted, shares)
    weeks = dealt[Split._fields.index("validation")]
    if len(weeks) % count:
        raise Refusal(
            f"les {len(weeks)} setmanes de validació no fan {count} parts iguals"
        )
    return [_days_of(days, chunk) for chunk in np.split(weeks, count)]
