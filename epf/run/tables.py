"""Els números d'una passada, en fitxers d'un sol esquema.

Es fan dels fitxers que la passada va desar, i van al costat de les figures.
"""

import json

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import measures, metrics
from epf.contract.record import fitted_files
from epf.parameters import HOURS_PER_DAY
from epf.run import families

PART_I_MAIN = "part_i_main.csv"
PART_I_SECONDARY = "part_i_secondary.csv"
PART_II_MAIN = "part_ii_main.csv"

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


def _best(where, complete):
    """Torna la millor configuració de cada seed de les famílies completes, `complete`.

    De cada una, els dies de prova, `days`, i la predicció i el preu de cada un,
    `predicted` i `actual`, `(dies, 24)`, i el que el fitxer porti de `training_mae` i
    `training_mse`. Falla si les files d'un fitxer no són les dels dies que diu.
    """
    found = {}
    for family in complete:
        found[family] = {}
        for (seed, rank), path in fitted_files(where / family).items():
            if rank:
                continue
            with np.load(path) as model:
                days = model["days_test"]
                if model["predicted"].size != len(days) * HOURS_PER_DAY:
                    raise ValueError(
                        f"{paths.relative(path)}: {model['predicted'].size} files per "
                        f"a {len(days)} dies de prova"
                    )
                found[family][seed] = {
                    "days": days,
                    "days_training": model["days_training"],
                    "days_validation": model["days_validation"],
                    "predicted": model["predicted"].reshape(len(days), -1),
                    "actual": model["actual"].reshape(len(days), -1),
                } | {
                    name: model[name].item()
                    for name in ("training_mae", "training_mse")
                    if name in model.files
                }
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
    """Escriu les files d'un fitxer de números, sencer o gens, i en torna la taula."""
    rows = _ordered(rows)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for name in INTEGERS:
        frame[name] = frame[name].astype("Int64")
    # Un recompte, que amb els decimals pandas escriuria com un d'ells, `3.0`.
    frame["value"] = pd.Series([row["value"] for row in rows], dtype=object)
    paths.write_whole(path, frame.to_csv(index=False, lineterminator="\n"))
    return frame


def write(where, main_only=False):
    """Escriu els números de la passada `where` als seus fitxers, i els torna.

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

    `part_i_secondary.csv`, les secundàries, de la millor configuració de cada seed de
    cada família completa: `rmse`, `mse`, `mdae`, `mean_error`, `wape`, `corr`, `wmae`,
    `cov_e`, `corr_f`, `mhd` i `mpd`; `rmae` sobre cada còpia que en dona un de
    secundari; `deviation_from_best`, i `deviation_from_best_mean`, `seed` buida;
    `mae` i `r2` de cada hora; sobre la còpia principal, `mase` i `rmsse`, i `rmae` de
    cada hora, `rmae_year` de cada any i `rmae_decile` de cada decil del preu, de l'1 al
    10, a `subject`, també sobre les seeds, `seed` buida; `price_decile_top`, el límit
    de dalt de cada decil, `family` buida; de cada família amb el test de Ljung-Box,
    `ljung_box_q`, `ljung_box_p` i `ljung_box_days` de la sèrie cosida de totes les
    seeds, `seed` buida, i de cada seed, i `autocorrelation` de la cosida a cada retard,
    a `subject`; a cada seed i parell ordenat, `dm_day` i `dm_hour`, el valor p que
    `other_family` és més precisa que `family`; a cada seed i parell en l'ordre de les
    famílies, `error_correlation`; i, sobre les seeds, `seed` buida, de cada parell en
    aquest ordre, `t_test`, `t_statistic`, `t_left`, `t_rope` i `t_right`, sobre el MAE
    de `family` menys el d'`other_family`, `t_left` per a `family` més precisa.

    `part_ii_main.csv`, les mesures principals de la Part II, de cada família
    interpretada completa, cada seed la mitjana dels seus models amb alguna
    contribució, `rank` buida: `share` de cada grup i `slope` per GW de cada sèrie amb
    panell de dependència, a `subject`, i les seves `share_mean`, `share_std`,
    `slope_mean` i `slope_std`, `seed` buida; `sign` de cada `slope_mean` i
    `sign_against`, les seeds que no el mostren; `order_against` de cada parell de
    grups, `subject` el que la mitjana posa per sobre d'`other_subject`; de la família
    amb interaccions, `partner_strength` de cada sèrie, `subject`, amb cada grup,
    `other_subject`, a la millor configuració de cada seed, i `partner_strength_mean` i
    `partner_seeds`, les seeds on el grup és dels tres més forts, `seed` buida; de la
    família el registre de la qual en diu `selection`, `selection` de cada columna, a
    `subject`, a cada hora i, `hour` buida, a totes; i `models_without_contribution` de
    cada seed.

    Torna la taula de cada fitxer pel seu nom. Amb `main_only`, el d'una passada que
    s'atura abans de les figures, només escriu `part_i_main.csv`.
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
    numbers = {
        PART_I_MAIN: _written(
            where / PART_I_MAIN, metrics.part_i_main(scores, denominator, complete)
        )
    }
    if main_only:
        return numbers
    best = _best(where, complete)
    numbers[PART_I_SECONDARY] = _written(
        where / PART_I_SECONDARY,
        metrics.part_i_secondary(best, records) + metrics.part_i_tests(best),
    )
    models = {}
    for family in complete:
        if not records[family].get("interpreted"):
            continue
        models[family] = {}
        for key, path in fitted_files(where / family).items():
            with np.load(path) as model:
                models[family][key] = measures.summarised(
                    model, records[family].get("selection")
                )
    numbers[PART_II_MAIN] = _written(
        where / PART_II_MAIN, measures.part_ii_main(models, records)
    )
    return numbers
