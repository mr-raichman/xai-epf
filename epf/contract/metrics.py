"""Les puntuacions de tots els models, i les mètriques de la Part I que se'n fan."""

import math

import numpy as np

from epf import Refusal
from epf.contract.explanation import PRICE_UNITS, UNITLESS

# Les unitats de cada mètrica principal.
MAIN_UNITS = {"r2": UNITLESS, "mae": PRICE_UNITS, "rmae": UNITLESS}


def _fail_on_absent(actual):
    """Falla si el preu real té cap valor absent, i diu quants.

    Un NaN faria tornar NaN la mesura sencera, o la faria saltar l'hora i informar-la
    com si fos neta, segons com es mitjanés.
    """
    absent = int(np.isnan(np.asarray(actual)).sum())
    # Només un error del codi hi arriba, i n'hi ha prou amb el recompte: el dataset no
    # té cap preu absent dins del seu tram, i `matrix.series` retalla cada sèrie al seu.
    if absent:
        raise ValueError(
            "el preu real té 1 valor absent"
            if absent == 1
            else f"el preu real té {absent} valors absents"
        )


def mae(predicted, actual):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def scored(predicted, actual):
    """Torna l'R² i el MAE de totes les files juntes.

    Falla amb un preu real absent, i refusa una predicció que no és un número finit: un
    sol NaN tornaria NaN totes dues puntuacions sense dir-ho.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    _fail_on_absent(actual)
    wrong = int((~np.isfinite(predicted)).sum())
    if wrong:
        raise Refusal(
            "la predicció té 1 valor no finit"
            if wrong == 1
            else f"la predicció té {wrong} valors no finits"
        )
    error = predicted - actual
    spread = actual - np.mean(actual)
    return {
        "r2": float(1 - np.sum(error**2) / np.sum(spread**2)),
        "mae": mae(predicted, actual),
    }


def sample_spread(values):
    """Torna la desviació estàndard mostral d'uns valors, la que divideix per n - 1.

    D'un sol valor no n'hi ha, i torna NaN sense cap avís.
    """
    if len(values) < 2:
        return math.nan
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def part_i_main(scores, denominator, complete):
    """Torna les files de les mètriques principals de la Part I, en l'ordre del fitxer.

    `scores` dona l'R² i el MAE de cada model ajustat, `{"r2": ..., "mae": ...}`, per
    família, en l'ordre de les famílies, i per `(seed, configuració)`; `denominator`,
    la família que divideix l'rMAE, o `None` si la passada no en té; i `complete`, les
    famílies completes. L'rMAE d'un model és el seu MAE sobre el de la millor
    configuració de `denominator` a la mateixa seed, i sense aquest no en té. Cada
    família completa es resumeix per la mitjana i la desviació mostral de cada mètrica
    sobre la millor configuració de cada seed, i una mètrica a la qual falta alguna
    seed no es resumeix. Cada fila és un diccionari amb les claus que s'hi apliquen.
    """
    base = scores.get(denominator, {})
    found = {name: {} for name in MAIN_UNITS}
    for family, fits in scores.items():
        for name in found:
            found[name][family] = {}
        for (seed, rank), scored in fits.items():
            found["r2"][family][seed, rank] = scored["r2"]
            found["mae"][family][seed, rank] = scored["mae"]
            if (seed, 0) in base:
                found["rmae"][family][seed, rank] = scored["mae"] / base[seed, 0]["mae"]
    rows = []
    for name, by_family in found.items():
        for family, values in by_family.items():
            rows += [
                {
                    "measure": name,
                    "family": family,
                    **({"other_family": denominator} if name == "rmae" else {}),
                    "seed": seed,
                    "rank": rank,
                    "value": value,
                    "unit": MAIN_UNITS[name],
                }
                for (seed, rank), value in values.items()
            ]
    for name, by_family in found.items():
        for family in by_family:
            seeds = sorted({seed for seed, _ in scores[family]})
            if family not in complete or any(
                (seed, 0) not in by_family[family] for seed in seeds
            ):
                continue
            best = [by_family[family][seed, 0] for seed in seeds]
            for summary, value in (
                ("mean", float(np.mean(best))),
                ("std", sample_spread(best)),
            ):
                rows.append(
                    {
                        "measure": f"{name}_{summary}",
                        "family": family,
                        **({"other_family": denominator} if name == "rmae" else {}),
                        "rank": 0,
                        "value": value,
                        "unit": MAIN_UNITS[name],
                    }
                )
    return rows
