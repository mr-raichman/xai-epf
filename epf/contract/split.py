"""La partició de Trebbien et al. (2023): les setmanes del període, barrejades per
llavor i repartides en entrenament, validació i prova.

A diferència del seu article, el preu del dia abans és una entrada, i per això
l'entrenament perd el dia de cada costat d'una setmana retinguda. La llegeixen la
passada i la cerca, que la passada llegeix, i per això és aquí i no a cap de les dues.
"""

import collections

import numpy as np
import pandas as pd

from epf.parameters import DAYS_PER_WEEK, GAP, SHARES

# Els dies de cada part d'una partició, en l'ordre en què es reparteixen les setmanes.
Split = collections.namedtuple("Split", "training validation test")

# Parts en què es parteix la validació perquè l'entrenament s'aturi quan qualsevol
# d'elles deixa de millorar, les de Trebbien et al. (2023).
VALIDATION_PARTS = 4


def _deal(seed, period, deleted, shares):
    """Els dies del període i, per a cada part, les setmanes que li toquen, en l'ordre
    en què la llavor les ha repartit.

    El període es talla en setmanes de set dies des del primer, i la setmana `deleted`
    en surt i no és de cap part. La resta es barregen amb la llavor i es reparteixen en
    l'ordre de `Split`: a la validació, la seva part de `shares` arrodonida a un
    múltiple de `VALIDATION_PARTS`, perquè es parteixi en parts iguals; a la prova, la
    seva arrodonida; i a l'entrenament, la resta.
    """
    days = pd.date_range(*period, freq="D")
    weeks, remainder = divmod(len(days), DAYS_PER_WEEK)
    if remainder:
        raise ValueError(f"{len(days)} dies no són setmanes senceres")
    first = days.get_indexer([pd.Timestamp(deleted[0])])[0]
    if (
        first < 0
        or first % DAYS_PER_WEEK
        or (pd.Timestamp(deleted[1]) - pd.Timestamp(deleted[0])).days
        != DAYS_PER_WEEK - 1
    ):
        raise ValueError(f"{deleted} no és cap de les setmanes del període")
    kept = np.delete(np.arange(weeks), first // DAYS_PER_WEEK)
    order = np.random.default_rng(seed).permutation(kept)
    _, validation_share, test_share = shares
    validation = VALIDATION_PARTS * round(
        len(kept) * validation_share / 100 / VALIDATION_PARTS
    )
    test = round(len(kept) * test_share / 100)
    ends = np.cumsum([len(kept) - validation - test, validation, test])
    return days, [order[start:end] for start, end in zip([0, *ends[:-1]], ends)]


def _days_of(days, weeks):
    """Els dies de les setmanes donades, en ordre de data."""
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
    """Els dies d'entrenament, de validació i de prova d'una llavor.

    Després de repartir les setmanes surt de l'entrenament tot dia que tingui un dia
    retingut a `gap` dies o menys, a qualsevol costat: aquest dia no és de cap part. La
    setmana suprimida no és retinguda, i els dies del seu costat es reparteixen com
    qualsevol altre.
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
    """Les setmanes de validació d'una llavor en `count` parts iguals, en l'ordre en
    què la llavor les ha repartit, de manera que cada part és de setmanes a l'atzar."""
    days, dealt = _deal(seed, period, deleted, shares)
    weeks = dealt[Split._fields.index("validation")]
    if len(weeks) % count:
        raise ValueError(
            f"{len(weeks)} setmanes de validació no fan {count} parts iguals"
        )
    return [_days_of(days, chunk) for chunk in np.split(weeks, count)]
