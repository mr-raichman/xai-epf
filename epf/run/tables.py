"""Els números d'una passada, en fitxers d'un sol esquema.

Es fan dels fitxers que la passada va desar, i van al costat de les figures.
"""

import json

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import metrics
from epf.contract.record import fitted_files
from epf.run import families

PART_I_MAIN = "part_i_main.csv"

# Les columnes de tots els fitxers, i les que són enters, que amb una casella buida
# pandas escriuria com a decimals, `3.0`.
COLUMNS = (
    "measure",
    "family",
    "other_family",
    "seed",
    "rank",
    "hour",
    "subject",
    "other_subject",
    "value",
    "unit",
)
INTEGERS = ("seed", "rank", "hour")


def _scores(where):
    """Torna l'R² i el MAE de cada model ajustat d'una passada.

    Per família, en l'ordre de les famílies, i per `(seed, configuració)`, de les que en
    tenen algun.
    """
    found = {}
    for family in families.EVERY:
        fits = {}
        for key, path in fitted_files(where / family).items():
            with np.load(path) as model:
                fits[key] = {name: model[name].item() for name in ("r2", "mae")}
        if fits:
            found[family] = fits
    return found


def _ordered(rows):
    """Torna les files per mesura, en l'ordre en què surten, família, seed, lloc i hora.

    Una clau buida va abans que les plenes, i l'ordre del que es mesura és el d'entrada.
    """
    measures = list(dict.fromkeys(row["measure"] for row in rows))

    def key(row):
        family = row.get("family")
        return (
            measures.index(row["measure"]),
            -1 if family is None else families.EVERY.index(family),
            *(-1 if row.get(name) is None else row[name] for name in INTEGERS),
        )

    return sorted(rows, key=key)


def _written(path, rows):
    """Escriu les files d'un fitxer de números, sencer o gens."""
    frame = pd.DataFrame(_ordered(rows), columns=COLUMNS)
    for name in INTEGERS:
        frame[name] = frame[name].astype("Int64")
    paths.write_whole(path, frame.to_csv(index=False, lineterminator="\n"))


def write(where):
    """Escriu els números de la passada `where` als seus fitxers.

    Cada fitxer té una fila per número, amb les columnes de `COLUMNS`: la mesura; la
    família i, d'una mesura entre dues, l'altra; la seed i la configuració del model,
    pel seu lloc a la cerca de la seed; l'hora, de 0 a 23, la del codi; el que es
    mesura i, d'una mesura entre dos, l'altre; el valor, amb tota la precisió, i la seva
    unitat. Una clau que no s'hi aplica queda buida: `seed`, en un número sobre totes
    les seeds; `rank` és 0 en un de la millor configuració d'una seed o en la seva
    mitjana sobre les seeds, i queda buida en un de totes les configuracions d'una seed
    juntes; i `family`, en un de cap família. Un valor que no pot existir, com la
    desviació d'una sola seed, queda buit en una fila que s'escriu igual. pandas llegeix
    els valors exactes amb `float_precision="round_trip"`, i no amb el seu lector per
    defecte, que en mou les darreres xifres.

    `part_i_main.csv`, les mètriques principals de la Part I: `r2`, `mae` i `rmae` de
    cada model ajustat que hi ha, `rmae` sobre la família el registre de la qual en diu
    `"main"` el `denominator`, `other_family`; i, de cada família completa, `r2_mean`,
    `r2_std`, `mae_mean`, `mae_std`, `rmae_mean` i `rmae_std`, sobre la millor
    configuració de cada seed, `seed` buida.
    """
    scores = _scores(where)
    complete = [family for family in scores if families.complete(where / family)]
    records = {
        family: json.loads(
            (where / family / paths.FAMILY_RECORD).read_text(encoding="utf-8")
        )
        for family in scores
    }
    # Una passada començada abans que els registres ho diguessin no té rMAE.
    denominator = next(
        (
            family
            for family, found in records.items()
            if found.get("denominator") == "main"
        ),
        None,
    )
    _written(where / PART_I_MAIN, metrics.part_i_main(scores, denominator, complete))
