"""El contracte que qualsevol conjunt de dades ha de complir.

El crida el carregador del conjunt i no qui el fa servir, de manera que no hi ha cap
camí que se'l salti: `price.load()` torna un marc que l'ha passat o aixeca. Aquí no
hi ha res d'OMIE ni d'ESIOS, que això és de cada font i es comprova a `audit.py`.
"""
import numpy as np
import pandas as pd

from epf.parameters import HOURS_PER_DAY


def refuse_incoherent(name, frame, gaps=()):
    """El contracte de qualsevol conjunt de dades. Torna el marc perquè es compongui.

    Cinc condicions, i cap d'elles és de cap font en concret: un dia repetit, un dia
    fora d'ordre, un dia que falta al mig, una graella que no és d'un dia sencer o un
    valor absent. Totes cinc produeixen un número plausible més endavant en lloc de
    petar, que és per això que es comproven aquí i no es deixen al model.

    `gaps` són les hores que el conjunt declara que la font no publica i que no
    s'omplen: un valor absent només s'admet en una d'elles.
    """
    def refuse(condition, detail):
        if condition:
            raise ValueError(f"el conjunt {name} no és coherent: {detail}")

    refuse(not frame.index.is_unique,
           f"{len(frame) - frame.index.nunique()} dies repetits a l'índex")
    refuse(not frame.index.is_monotonic_increasing, "l'índex no està ordenat")
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="D")
    missing = expected.difference(frame.index)
    refuse(len(missing) > 0,
           f"{len(missing)} dies buits entre {frame.index[0].date()} i "
           f"{frame.index[-1].date()}, el primer {missing[0].date() if len(missing) else ''}")
    refuse(frame.shape[1] != HOURS_PER_DAY,
           f"{frame.shape[1]} columnes i no {HOURS_PER_DAY}")
    days, hours = np.nonzero(frame.isna().to_numpy())
    absent = (frame.index[days] + pd.to_timedelta(hours, unit="h")).difference(
        pd.DatetimeIndex(gaps))
    refuse(len(absent) > 0,
           f"{len(absent)} valors absents, el primer a {absent[0] if len(absent) else ''}")
    return frame
