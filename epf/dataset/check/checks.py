"""Les comprovacions del conjunt de dades, les mateixes per a totes les sèries.

L'error de cada una donaria més endavant un número plausible en lloc de fallar: un dia i
una hora repetits doblarien el seu pes, unes files fora d'ordre o un dia que falta
desplaçarien els retards cap a un dia que no és el que toca, un dia sense les seves 24
hores desalinearia les hores, i un valor absent es propagaria al model.
"""

import numpy as np
import pandas as pd

from epf import Refusal
from epf.parameters import HOURS_PER_DAY


def check(table, gaps):
    """Refusa el conjunt de dades si no és coherent.

    `table` té una fila per dia i hora, amb el dia, `AAAA-MM-DD`, i l'hora, de 0 a 23,
    com a índex, i una columna per sèrie. `gaps` són, per sèrie, les hores que la font
    no va publicar: un valor absent només s'admet en una d'elles, i es busca dins del
    tram de la sèrie, del primer al darrer dia amb algun valor, perquè fora d'ell la
    sèrie no té res.
    """

    def refused(detail):
        return Refusal(f"el conjunt de dades no és coherent: {detail}")

    def counted(count, one, many, first=None):
        """Torna `3 dies sense 24 hores, el primer 2020-03-29`, en singular o en plural.

        `one` i `many` són el nom en singular i en plural, i sense `first` no en diu el
        primer.
        """
        found = f"1 {one}" if count == 1 else f"{count} {many}"
        if first is None:
            return found
        return f"{found}, el {first}" if count == 1 else f"{found}, el primer {first}"

    days = pd.to_datetime(table.index.get_level_values(0), format="%Y-%m-%d")
    hours = table.index.get_level_values(1)
    repeated = int(table.index.duplicated().sum())
    if repeated:
        raise refused(
            counted(
                repeated,
                "fila repeteix un dia i una hora",
                "files repeteixen un dia i una hora",
            )
        )
    if not table.index.is_monotonic_increasing:
        raise refused("les files no estan ordenades")
    present = days.unique()
    missing = pd.date_range(present[0], present[-1], freq="D").difference(present)
    if len(missing):
        span = f"de {present[0].date()} a {present[-1].date()}"
        raise refused(
            counted(
                len(missing),
                f"dia sense cap fila {span}",
                f"dies sense cap fila {span}",
                missing[0].date(),
            )
        )
    outside = int((~np.isin(hours, np.arange(HOURS_PER_DAY))).sum())
    if outside:
        wrong_hour = f"una hora fora de 0 a {HOURS_PER_DAY - 1}"
        raise refused(
            counted(outside, f"fila amb {wrong_hour}", f"files amb {wrong_hour}")
        )
    counts = days.value_counts()
    short = counts.index[counts != HOURS_PER_DAY].sort_values()
    if len(short):
        raise refused(
            counted(
                len(short),
                f"dia sense {HOURS_PER_DAY} hores",
                f"dies sense {HOURS_PER_DAY} hores",
                short[0].date(),
            )
        )

    instants = days + pd.to_timedelta(hours, unit="h")
    for name in table.columns:
        values = table[name].to_numpy()
        found = np.flatnonzero(~np.isnan(values))
        if not len(found):
            raise refused(f"la sèrie {name} no té cap valor")
        span = (days >= days[found[0]]) & (days <= days[found[-1]])
        absent = instants[span & np.isnan(values)].difference(gaps[name])
        if len(absent):
            moment = f"{absent[0]:%Y-%m-%d %H:%M}"
            raise refused(
                f"la sèrie {name} té 1 valor absent, a {moment}"
                if len(absent) == 1
                else f"la sèrie {name} té {len(absent)} valors absents, el primer a "
                f"{moment}"
            )
