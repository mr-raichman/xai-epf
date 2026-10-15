"""Les comprovacions del conjunt de dades, les mateixes per a totes les sèries, sobre el
conjunt final.

Cap d'elles no és de cap font, i totes produirien un número plausible més endavant en
lloc de petar, que és per això que es comproven aquí i no es deixen al model: un dia i una
hora repetits doblarien el seu pes, unes files fora d'ordre o un dia que falta desplaçarien
els retards cap a un dia que no és el que toca, un dia sense les seves 24 hores
desalinearia les hores, i un valor absent es propagaria al model.
"""
import numpy as np
import pandas as pd

from epf.parameters import HOURS_PER_DAY


def check(table, gaps):
    """Refusa el conjunt de dades si no és coherent.

    `table` té una fila per dia i hora, amb el dia, `AAAA-MM-DD`, i l'hora, de 0 a 23, com
    a índex, i una columna per sèrie. `gaps` són, per sèrie, les hores que declara que la
    font no va publicar, que no s'omplen: un valor absent només s'admet en una d'elles, i
    es busca dins del tram de la sèrie, del primer al darrer dia amb algun valor, perquè
    fora d'ell la sèrie no té res.
    """
    def refuse(condition, detail):
        if condition:
            raise ValueError(f"el conjunt de dades no és coherent: {detail}")

    days = pd.to_datetime(table.index.get_level_values(0), format="%Y-%m-%d")
    hours = table.index.get_level_values(1)
    refuse(table.index.has_duplicates,
           f"{table.index.duplicated().sum()} dies i hores repetits")
    refuse(not table.index.is_monotonic_increasing, "les files no estan ordenades")
    present = days.unique()
    missing = pd.date_range(present[0], present[-1], freq="D").difference(present)
    refuse(len(missing) > 0,
           f"{len(missing)} dies buits entre {present[0].date()} i {present[-1].date()}, "
           f"el primer {missing[0].date() if len(missing) else ''}")
    outside = ~np.isin(hours, np.arange(HOURS_PER_DAY))
    refuse(outside.any(), f"{outside.sum()} hores fora de 0 a {HOURS_PER_DAY - 1}")
    counts = days.value_counts()
    short = counts.index[counts != HOURS_PER_DAY].sort_values()
    refuse(len(short) > 0, f"{len(short)} dies sense {HOURS_PER_DAY} hores, el primer "
                           f"{short[0].date() if len(short) else ''}")

    instants = days + pd.to_timedelta(hours, unit="h")
    for name in table.columns:
        values = table[name].to_numpy()
        found = np.flatnonzero(~np.isnan(values))
        refuse(len(found) == 0, f"la sèrie {name} no té cap valor")
        span = (days >= days[found[0]]) & (days <= days[found[-1]])
        absent = instants[span & np.isnan(values)].difference(gaps[name])
        refuse(len(absent) > 0, f"la sèrie {name} té {len(absent)} valors absents, el primer "
                                f"a {absent[0] if len(absent) else ''}")
