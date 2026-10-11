"""La part B: el protocol de Trebbien et al. (2023) sobre el període de desenvolupament.

Les setmanes es barregen per llavor i es reparteixen en entrenament, validació i
prova, i la robustesa és la dispersió entre llavors. A diferència del seu article, el
preu del dia abans és una entrada, i per això l'entrenament perd el dia de cada costat
d'una setmana retinguda.
"""
import collections

import numpy as np
import pandas as pd

from epf.parameters import DAYS_PER_WEEK, GAP, SHARES

# Els dies de cada part d'una partició, en l'ordre en què es reparteixen les setmanes.
Split = collections.namedtuple("Split", "training validation test")


def split(seed, period, shares=SHARES, gap=GAP):
    """Els dies d'entrenament, de validació i de prova d'una llavor.

    El període es talla en setmanes de set dies des del primer, que es barregen amb la
    llavor i es reparteixen en l'ordre de `Split` segons els percentatges de `shares`.
    Després surt de l'entrenament tot dia que tingui un dia retingut a `gap` dies o
    menys, a qualsevol costat: aquest dia no és de cap part.
    """
    days = pd.date_range(*period, freq="D")
    weeks, remainder = divmod(len(days), DAYS_PER_WEEK)
    if remainder:
        raise ValueError(f"{len(days)} dies no són setmanes senceres")
    if sum(shares) != 100 or any(weeks * share % 100 for share in shares):
        raise ValueError(f"{shares} no reparteix {weeks} setmanes senceres")

    order = np.random.default_rng(seed).permutation(weeks)
    part_of_week = np.empty(weeks, dtype=int)
    ends = np.cumsum([weeks * share // 100 for share in shares])
    for part, (start, end) in enumerate(zip([0, *ends[:-1]], ends)):
        part_of_week[order[start:end]] = part
    part_of_day = np.repeat(part_of_week, DAYS_PER_WEEK)

    training = part_of_day == 0
    beside = np.zeros(len(days), dtype=bool)
    for offset in range(1, gap + 1):
        beside[offset:] |= ~training[:-offset]
        beside[:-offset] |= ~training[offset:]
    return Split(days[training & ~beside], days[part_of_day == 1], days[part_of_day == 2])
