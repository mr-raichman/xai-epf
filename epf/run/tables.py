"""La taula de les puntuacions d'una passada.

Es fa dels fitxers que la passada va desar, i va al costat de les figures.
"""

import json

import numpy as np
import pandas as pd

from epf import paths
from epf.contract.figure import importances
from epf.contract.record import fitted_files
from epf.run import families

# Les puntuacions que porta un model ajustat, les de la distribució només a les famílies
# que en prediuen una, i la taula on van.
SCORE_NAMES = ("r2", "mae", "crps", "kupiec_50", "kupiec_90")
SCORES = "scores.csv"


def scores(where):
    """Escriu a `scores.csv` les puntuacions de cada model ajustat d'una passada.

    Una fila per família, seed i configuració, en l'ordre de les famílies, i buida la
    puntuació que una família no té. La família el registre de la qual diu una columna
    de comptador de dies, la sonda de datació, hi porta també la seva lectura,
    `counter_share`: la part del comptador en la importància de les columnes.
    """
    rows = []
    for family in families.EVERY:
        files = fitted_files(where / family)
        if not files:
            continue
        record = where / family / paths.FAMILY_RECORD
        counter = json.loads(record.read_text(encoding="utf-8")).get("counter")
        for (seed, rank), path in files.items():
            with np.load(path) as found:
                row = {"family": family, "seed": seed, "rank": rank} | {
                    name: found[name].item() for name in SCORE_NAMES if name in found
                }
                if counter is not None:
                    by_column, _ = importances(found["values"], found["groups"])
                    column = list(found["columns"]).index(counter)
                    row["counter_share"] = by_column[column]
            rows.append(row)
    frame = pd.DataFrame(
        rows, columns=["family", "seed", "rank", *SCORE_NAMES, "counter_share"]
    )
    # Amb algun buit, una columna de recomptes seria de coma flotant, i s'hi escriuria
    # `23.0`.
    counts = ["kupiec_50", "kupiec_90"]
    frame[counts] = frame[counts].astype("Int64")
    frame.to_csv(where / SCORES, index=False, lineterminator="\n", encoding="utf-8")
