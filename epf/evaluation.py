"""El CRPS d'una predicció distribucional, i la guarda que ha de passar el preu real.

Res d'aquí tria cap paràmetre: escollir es fa sobre la validació, no sobre la prova.
"""
import numpy as np

# Els percentils sobre els quals s'aproxima el CRPS. Marcjasz et al. (2023) en fan
# servir 99, que és el pinball mitjà sobre tots els centils interiors.
PERCENTILES = np.arange(1, 100) / 100


def refuse_missing(frame, what):
    """Falla si la taula porta cap valor absent, anomenant on és el primer.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés: es refusa abans, dient on és.
    """
    missing = frame.isna()
    if not missing.to_numpy().any():
        return
    flat = missing.stack()
    row, column = next(iter(flat[flat].index))
    raise ValueError(f"{int(missing.to_numpy().sum())} valors absents a {what}, "
                     f"el primer a {row} hora {column}")


def crps(actual, quantiles, probabilities=PERCENTILES):
    """CRPS d'una predicció distribucional, pel pinball mitjà sobre els percentils.

    El CRPS és la mesura pròpia d'una predicció distribucional: diu si tota la
    distribució és bona i no només el seu centre, de manera que un model que encerta el
    preu i s'equivoca en la incertesa en paga el preu. S'aproxima com fa Marcjasz et
    al. (2023), amb el pinball mitjà sobre 99 percentils, que és l'aproximació
    habitual.

    El pinball d'un quantil `q` és `(1-q)(Q - y)` quan el preu real queda per sota del
    quantil predit i `q(y - Q)` quan queda per sobre: penalitza més quedar-se curt als
    quantils alts i llarg als baixos, que és el que fa que una distribució massa
    estreta surti pitjor que una massa ampla amb el mateix centre.

    `quantiles` és `(dies, hores, percentils)` i `actual` la taula de preus dels
    mateixos dies, en el mateix ordre.
    """
    refuse_missing(actual, "el preu real")
    if not np.isfinite(quantiles).all():
        raise ValueError("els quantils predits porten valors no finits")
    difference = actual.to_numpy()[:, :, None] - quantiles
    return float(np.maximum(probabilities * difference,
                            (probabilities - 1) * difference).mean())
