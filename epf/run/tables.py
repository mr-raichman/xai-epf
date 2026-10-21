"""La taula de les puntuacions d'una passada.

Es fa dels fitxers que la passada va desar, i va al costat de les figures.
"""

import numpy as np
import pandas as pd

from epf.contract.record import fitted_files
from epf.models.lightgbm.hyperparameters import DAY_COUNTER
from epf.run import families

# Les puntuacions que porta un model ajustat, les de la distribució només a les famílies
# que en prediuen una, i la taula on van.
SCORES = ("r2", "mae", "crps", "kupiec_50", "kupiec_90")
SCORES_FILE = "scores.csv"

# La família de la sonda de datació.
PROBE = "date_probe"


def scores(where):
    """Escriu a `scores.csv` les puntuacions de cada model ajustat d'una passada.

    Una fila per família, seed i configuració, en l'ordre de les famílies, i buida la
    puntuació que una família no té. La sonda de datació hi porta també la seva lectura,
    `counter_share`: la part del comptador de dies en la mitjana del valor absolut de
    les contribucions.
    """
    rows = []
    for family in families.EVERY:
        for (seed, rank), path in fitted_files(where / family).items():
            with np.load(path) as found:
                row = {"family": family, "seed": seed, "rank": rank} | {
                    name: found[name].item() for name in SCORES if name in found
                }
                if family == PROBE:
                    weight = np.abs(found["values"].astype(np.float64)).mean(axis=0)
                    counter = list(found["columns"]).index(DAY_COUNTER)
                    row["counter_share"] = weight[counter] / weight.sum()
            rows.append(row)
    frame = pd.DataFrame(
        rows, columns=["family", "seed", "rank", *SCORES, "counter_share"]
    )
    # Amb algun buit, una columna de recomptes seria de coma flotant, i s'hi escriuria
    # `23.0`.
    counts = ["kupiec_50", "kupiec_90"]
    frame[counts] = frame[counts].astype("Int64")
    frame.to_csv(
        where / SCORES_FILE, index=False, lineterminator="\n", encoding="utf-8"
    )
