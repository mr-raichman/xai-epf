"""Les taules d'una passada: les puntuacions, i la lectura de la sonda de datació.

Es fan dels fitxers que la passada va desar, i van al costat de les figures.
"""

import numpy as np
import pandas as pd

from epf.contract.record import fitted_files
from epf.models.lightgbm.date_probe import DAY_COUNTER
from epf.models.lightgbm.hyperparameters import NAME as TREE
from epf.run import families

# Les puntuacions que porta un model ajustat, les de la distribució només a les famílies
# que en prediuen una, i la taula on van.
SCORES = ("r2", "mae", "crps", "kupiec_50", "kupiec_90")
SCORES_FILE = "scores.csv"

# La família de la sonda, i la taula de la seva lectura.
PROBE = "date_probe"
PROBE_FILE = "date_probe.csv"


def _written(rows, columns, path, counts=()):
    """Escriu una taula, amb les columnes de recomptes, `counts`, en enters.

    Una columna amb algun buit seria de coma flotant, i els recomptes s'hi escriurien
    `23.0`.
    """
    frame = pd.DataFrame(rows, columns=columns)
    frame[list(counts)] = frame[list(counts)].astype("Int64")
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def scores(where):
    """Escriu a `scores.csv` les puntuacions de cada model ajustat d'una passada.

    Una fila per família, seed i configuració, en l'ordre de les famílies, i buida la
    puntuació que una família no té.
    """
    rows = []
    for family in families.EVERY:
        for (seed, rank), path in fitted_files(where / family).items():
            with np.load(path) as found:
                rows.append(
                    {"family": family, "seed": seed, "rank": rank}
                    | {name: found[name].item() for name in SCORES if name in found}
                )
    _written(
        rows,
        ["family", "seed", "rank", *SCORES],
        where / SCORES_FILE,
        counts=("kupiec_50", "kupiec_90"),
    )


def date_probe(where):
    """Escriu a `date_probe.csv` la lectura de la sonda de datació de cada seed.

    La part del comptador de dies en la mitjana del valor absolut de les contribucions,
    i el MAE de la sonda i el de la millor configuració de l'arbre, sobre les mateixes
    files de prova. Una seed a la qual falta la sonda o l'arbre no hi és.
    """
    trees = fitted_files(where / TREE)
    rows = []
    for (seed, _), path in fitted_files(where / PROBE).items():
        if (seed, 0) not in trees:
            continue
        with np.load(path) as probe, np.load(trees[seed, 0]) as tree:
            weight = np.abs(probe["values"].astype(np.float64)).mean(axis=0)
            counter = list(probe["columns"]).index(DAY_COUNTER)
            rows.append(
                {
                    "seed": seed,
                    "counter_share": weight[counter] / weight.sum(),
                    "mae_tree": tree["mae"].item(),
                    "mae_probe": probe["mae"].item(),
                }
            )
    _written(
        rows, ["seed", "counter_share", "mae_tree", "mae_probe"], where / PROBE_FILE
    )
