"""Les mesures de la Part II, de les contribucions que desa cada model ajustat explicat.

Cada seed hi compta una vegada, la mitjana de les seves configuracions, totes igual.
"""

import collections
import itertools
import math

import numpy as np
from scipy import stats

from epf import information
from epf.contract.explanation import CALENDAR, DATE, UNITLESS
from epf.contract.metrics import sample_spread
from epf.information import KINDS, SERIES
from epf.parameters import HOUR_COLUMNS, HOURS_PER_DAY

# MW per GW, perquè els pendents es llegeixen per GW, com fan Trebbien et al. (2023).
PER_GW = 1000

# Els grups que tenen panells de dependència: els de les sèries que l'arbre llegeix a
# l'hora de la fila, un valor per fila, les previsions com a Trebbien et al. (2023) i
# el programa igual, en MW o en MWh.
DEPENDENCE = {
    name
    for name, (series, _) in information.groups().items()
    if KINDS[SERIES[series].kind].at_the_hour
}

# Socis de cada sèrie a les interaccions, els que més hi interactuen: la regla que fa de
# les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3

# El grup del preu del dia abans, i les seves hores 22 a 24, comptades des de 0: el
# vespre que Uniejewski et al. (2016) troben més triat que l'hora mateixa.
PRICE = information.column(information.PRICE, 1)
LATE_HOURS = [21, 22, 23]

# Els grups primers que comparen dos acords de Krishna et al. (2024).
TOP = 3


def contributions(model):
    """Torna les contribucions d'un model ajustat, `(files, columnes)`.

    Les de la xarxa neuronal, `(dies, columnes, 24)`, amb una fila per dia i hora.
    """
    values = model["values"]
    if values.ndim == 2:
        return values
    return values.transpose(0, 2, 1).reshape(-1, values.shape[1])


def importances(values, groups):
    """Torna la importància de cada columna i la de cada grup, la seva part del total.

    És la mitjana del valor absolut de les contribucions, en part del total com la
    donen Pesenti i O'Sullivan (2026), de manera que la d'una columna i la d'un grup es
    llegeixen igual. La d'un grup suma les contribucions de les seves columnes abans de
    prendre-la, com Pesenti i O'Sullivan (2025). Es calcula en `float64`.
    """
    values = np.asarray(values, dtype=np.float64)
    by_column = np.abs(values).mean(axis=0)
    names = list(dict.fromkeys(groups))
    by_group = np.array(
        [np.abs(values[:, groups == name].sum(axis=1)).mean() for name in names]
    )
    return by_column / by_column.sum(), dict(
        zip(names, by_group / by_group.sum(), strict=True)
    )


def ordered(names):
    """Torna uns grups amb les sèries en ordre, i el calendari i el comptador al final.

    Les matrius barregen el calendari amb les sèries.
    """
    names = [str(name) for name in names]
    last = [name for name in (CALENDAR, DATE) if name in names]
    return [name for name in names if name not in last] + last


def present(model):
    """Torna els grups de `DEPENDENCE` que porta un model ajustat, en el seu ordre."""
    return [str(name) for name in dict.fromkeys(model["groups"]) if name in DEPENDENCE]


def members(model, name):
    """Torna les posicions de les columnes del grup `name` en un model ajustat.

    Les troba pel grup que la passada hi va desar: una a la matriu per hora, la de
    l'hora de la fila, i les 24 de la diària en ordre d'hora.
    """
    members = np.flatnonzero(model["groups"] == name)
    if len(members) == 1:
        return members
    hour = {column: number for number, column in enumerate(HOUR_COLUMNS)}
    return np.array(
        sorted(members, key=lambda j: hour[str(model["columns"][j]).rsplit("_", 1)[1]])
    )


def pairs(model, values, name):
    """Torna el valor del grup `name` i la seva contribució a l'hora de cada fila."""
    inputs = model["inputs"]
    found = members(model, name)
    if len(found) == 1:
        return inputs[:, found[0]], values[:, found[0]]
    # La xarxa neuronal: a la fila del dia d i l'hora h, la columna de l'hora h.
    x = inputs[:, found]
    y = values.reshape(len(inputs), len(HOUR_COLUMNS), -1)[
        :, np.arange(len(found)), found
    ]
    return x.ravel(), y.ravel()


def line(model, values, name):
    """Torna el pendent per GW i l'ordenada de la recta de la contribució de `name`."""
    x, y = (np.asarray(one, dtype=np.float64) for one in pairs(model, values, name))
    # En `float64`: amb els `float32` desats, `polyfit` ajusta en `float32`.
    slope, intercept = np.polyfit(x / PER_GW, y, 1)
    return float(slope), float(intercept)


def slope(model, values, name):
    """Torna el pendent per GW de la contribució del grup `name` contra el seu valor."""
    return line(model, values, name)[0]


def strengths(model):
    """Torna la força de cada grup amb cada sèrie de `DEPENDENCE` a les interaccions.

    De cada sèrie, la mitjana del valor absolut de la suma de les interaccions de les
    columnes de cada altre grup amb la seva, el doble, com la dibuixa la figura: SHAP
    reparteix la d'un parell a parts iguals entre les seves dues columnes.
    """
    groups = model["groups"]
    interactions = np.asarray(model["interactions"], dtype=np.float64)
    found = {}
    for name in present(model):
        (j,) = members(model, name)
        found[name] = {
            other: float(
                np.abs(2 * interactions[:, j, groups == other].sum(axis=1)).mean()
            )
            for other in ordered(dict.fromkeys(groups))
            if other != name
        }
    return found


def summarised(model, selection=False):
    """Torna el que les mesures llegeixen d'un model ajustat obert.

    `None` si no té cap contribució. Si no, la part de cada grup en la importància,
    `shares`, en l'ordre d'`ordered`; el pendent per GW de cada grup de `DEPENDENCE`,
    `slopes`; i la unitat de les contribucions, `unit`. Amb `selection`, les columnes,
    `columns`, i si les contribucions de cada una a cada hora prevista no són totes
    zero, `selected`, `(24, columnes)`. D'un fitxer amb interaccions, `strengths`,
    `partners`. La contribució absoluta mitjana de cada grup, `magnitudes`, i la de cada
    fila, `rows`, `(files, grups)`. Amb les 24 hores del preu del dia abans, la seva
    contribució absoluta mitjana a cada hora prevista, a la mateixa hora, `own_hour`, i
    de mitjana a les de `LATE_HOURS`, `late_evening`.
    """
    values = contributions(model)
    if not np.any(values):
        return None
    groups = model["groups"]
    _, shares = importances(values, groups)
    names = ordered(shares)
    every = np.asarray(values, dtype=np.float64)
    rows = np.stack([every[:, groups == name].sum(axis=1) for name in names], axis=1)
    found = {
        "shares": {name: float(shares[name]) for name in names},
        "slopes": {name: slope(model, values, name) for name in present(model)},
        "unit": str(model["unit"]),
        "magnitudes": dict(zip(names, np.abs(rows).mean(axis=0).tolist(), strict=True)),
        "rows": rows,
    }
    if np.sum(groups == PRICE) == HOURS_PER_DAY:
        hours = np.arange(len(every)) % HOURS_PER_DAY
        # A cada hora prevista, de cada hora del preu.
        heat = np.array(
            [
                np.abs(every[hours == hour][:, members(model, PRICE)]).mean(axis=0)
                for hour in range(HOURS_PER_DAY)
            ]
        )
        found["own_hour"] = np.diag(heat)
        found["late_evening"] = heat[:, LATE_HOURS].mean(axis=1)
    if selection:
        hours = np.tile(np.arange(HOURS_PER_DAY), len(values) // HOURS_PER_DAY)
        found["columns"] = [str(column) for column in model["columns"]]
        found["selected"] = np.array(
            [
                np.any(values[hours == hour] != 0, axis=0)
                for hour in range(HOURS_PER_DAY)
            ]
        )
    if "interactions" in model:
        found["partners"] = strengths(model)
    return found


def _by_seed(kept, kind, names):
    """Torna el valor de cada seed, la mitjana dels seus models, de cada nom de `names`.

    `kept` dona els models de cada seed amb alguna contribució, i `kind` què se'n
    llegeix, `shares` o `slopes`.
    """
    return {
        seed: {
            name: float(np.mean([one[kind][name] for one in models])) for name in names
        }
        for seed, models in kept.items()
    }


def part_ii_main(models, roles):
    """Torna les files de les mesures principals de la Part II.

    `models` dona, per família interpretada, en l'ordre de les famílies, i per `(seed,
    configuració)`, el que en torna `summarised`, o `None`; `roles`, el registre de cada
    família. El valor d'una seed és la mitjana dels seus models que tenen alguna
    contribució; una seed sense cap no en té, queda fora de les mitjanes i va contra
    cada troballa de la família, com una seed que no la mostra: un pendent de 0 o dos
    grups empatats.
    """
    rows = []

    def add(measure, family, value, unit=UNITLESS, **keys):
        rows.append(
            {"measure": measure, "family": family, **keys, "value": value, "unit": unit}
        )

    for family, fits in models.items():
        seeds = sorted({seed for seed, _ in fits})
        kept = {
            seed: [found for (one, _), found in fits.items() if one == seed and found]
            for seed in seeds
        }
        kept = {seed: found for seed, found in kept.items() if found}
        if kept:
            first = next(iter(kept.values()))[0]
            unit = first["unit"]
            shares = _by_seed(kept, "shares", first["shares"])
            slopes = _by_seed(kept, "slopes", first["slopes"])
            means = {"share": {}, "slope": {}}
            for measure, by_seed, names, measured_in in (
                ("share", shares, first["shares"], UNITLESS),
                ("slope", slopes, first["slopes"], f"{unit} per GW"),
            ):
                for name in names:
                    values = [by_seed[seed][name] for seed in kept]
                    means[measure][name] = float(np.mean(values))
                    for seed in kept:
                        add(
                            measure,
                            family,
                            by_seed[seed][name],
                            measured_in,
                            seed=seed,
                            subject=name,
                        )
                    add(
                        f"{measure}_mean",
                        family,
                        means[measure][name],
                        measured_in,
                        subject=name,
                    )
                    add(
                        f"{measure}_std",
                        family,
                        sample_spread(values),
                        measured_in,
                        subject=name,
                    )
            for name in first["slopes"]:
                sign = int(np.sign(means["slope"][name]))
                shown = sum(
                    sign != 0 and np.sign(slopes[seed][name]) == sign for seed in kept
                )
                add("sign", family, sign, subject=name)
                add("sign_against", family, len(seeds) - shown, subject=name)
            share = means["share"]
            for one, other in itertools.combinations(first["shares"], 2):
                higher, lower = (
                    (other, one) if share[other] > share[one] else (one, other)
                )
                shown = 0
                if share[one] != share[other]:
                    shown = sum(
                        shares[seed][higher] > shares[seed][lower] for seed in kept
                    )
                add(
                    "order_against",
                    family,
                    len(seeds) - shown,
                    subject=higher,
                    other_subject=lower,
                )
            strongest = {}
            for seed in kept:
                best = fits.get((seed, 0))
                if not best or "partners" not in best:
                    continue
                for name, by_group in best["partners"].items():
                    for group, value in by_group.items():
                        add(
                            "partner_strength",
                            family,
                            value,
                            unit,
                            seed=seed,
                            rank=0,
                            subject=name,
                            other_subject=group,
                        )
                    top = sorted(by_group, key=lambda group: -by_group[group])[
                        :PARTNERS
                    ]
                    strongest.setdefault(name, []).append((by_group, top))
            for name, found in strongest.items():
                for group in found[0][0]:
                    mean = float(np.mean([by_group[group] for by_group, _ in found]))
                    add(
                        "partner_strength_mean",
                        family,
                        mean,
                        unit,
                        rank=0,
                        subject=name,
                        other_subject=group,
                    )
                    count = sum(group in top for _, top in found)
                    add(
                        "partner_seeds",
                        family,
                        count,
                        rank=0,
                        subject=name,
                        other_subject=group,
                    )
            if roles[family].get("selection"):
                # Uniejewski et al. (2016): un coeficient zero, que la contribució de la
                # columna a l'hora sigui zero a totes les files.
                selected = np.mean(
                    [one["selected"] for models in kept.values() for one in models],
                    axis=0,
                )
                for number, column in enumerate(first["columns"]):
                    for hour in range(HOURS_PER_DAY):
                        add(
                            "selection",
                            family,
                            float(selected[hour, number]),
                            subject=column,
                            hour=hour,
                        )
                    add(
                        "selection",
                        family,
                        float(selected[:, number].mean()),
                        subject=column,
                    )
        for seed in seeds:
            count = sum(
                found is None for (one, _), found in fits.items() if one == seed
            )
            add("models_without_contribution", family, count, seed=seed)
    return rows


def counter_share(model, counter):
    """Torna la part de la columna `counter` en la importància de les d'un model."""
    by_column, _ = importances(contributions(model), model["groups"])
    return float(by_column[list(model["columns"]).index(counter)])


def stability(models):
    """Torna, de cada grup, la desviació entre els models d'una seed i la seva magnitud.

    `models` són els d'una seed amb alguna contribució, com els torna `summarised`, que
    expliquen les mateixes files. La desviació és la mitjana sobre les files de la
    desviació estàndard mostral entre els models de la contribució del grup a la fila,
    NaN amb un sol model, la de l'índex d'estabilitat de Caraker et al. (2026b); la
    magnitud, la contribució absoluta mitjana de tots.
    """
    stacked = np.stack([one["rows"] for one in models])
    names = list(models[0]["magnitudes"])
    deviation = (
        stacked.std(axis=0, ddof=1).mean(axis=0)
        if len(models) > 1
        else np.full(len(names), np.nan)
    )
    magnitude = np.abs(stacked).mean(axis=(0, 1))
    return {
        name: (float(one), float(other))
        for name, one, other in zip(names, deviation, magnitude, strict=True)
    }


def spearman_pairs(table):
    """Torna la correlació de Spearman de les parts dels grups de cada parell de models.

    `table` dona les parts de cada model d'una família amb alguna contribució,
    `(models, grups)`.
    """
    if len(table) < 2:
        return np.array([])
    found = np.atleast_2d(stats.spearmanr(table, axis=1).statistic)
    return found[np.triu_indices_from(found, k=1)]


def _order(shares):
    """Torna l'ordre dels grups per la seva part, `a > b`, i `a = b` si són iguals."""
    names = sorted(shares, key=lambda name: -shares[name])
    written = names[0]
    for before, name in itertools.pairwise(names):
        written += f" {'=' if shares[before] == shares[name] else '>'} {name}"
    return written


def _between_seeds(values, seeds):
    """Torna la part de la suma de quadrats d'uns valors que és entre seeds.

    La resta és dins de cada seed, entre les seves configuracions. Uns valors tots
    iguals tornen NaN.
    """
    values, seeds = np.asarray(values, dtype=np.float64), np.asarray(seeds)
    centre = values.mean()
    between = sum(
        (seeds == seed).sum() * (values[seeds == seed].mean() - centre) ** 2
        for seed in np.unique(seeds)
    )
    total = ((values - centre) ** 2).sum()
    # Uns valors tots iguals, com el pendent d'una sèrie que cap model no llegeix, no
    # tenen cap dispersió que es pugui partir.
    return float(between / total) if total else math.nan


def _agreements(first, second):
    """Torna els cinc acords de Krishna et al. (2024) entre dues famílies a una seed.

    Cada una és `(parts, pendents)`, les mitjanes de la seed en el mateix ordre de
    grups i de sèries: la correlació dels rànquings i l'acord dels parells sobre les
    parts, el dels `TOP` primers i el dels seus llocs, i el del signe dels pendents.
    """
    (shares, slopes), (other_shares, other_slopes) = first, second
    order = np.argsort(-shares, kind="stable")
    other_order = np.argsort(-other_shares, kind="stable")
    pairs = list(itertools.combinations(range(len(shares)), 2))
    return {
        "rank_correlation": float(stats.spearmanr(shares, other_shares).statistic),
        "pairwise_rank_agreement": float(
            np.mean(
                [
                    np.sign(shares[i] - shares[j])
                    == np.sign(other_shares[i] - other_shares[j])
                    for i, j in pairs
                ]
            )
        ),
        "feature_agreement": len(set(order[:TOP]) & set(other_order[:TOP])) / TOP,
        "rank_agreement": float(np.mean(order[:TOP] == other_order[:TOP])),
        "sign_agreement": float(np.mean(np.sign(slopes) == np.sign(other_slopes))),
    }


def part_ii_secondary(models, stabilities, counters):
    """Torna les files de les mesures secundàries de la Part II.

    `models` és com a `part_ii_main`, sense les files de cada model; `stabilities`
    dona, per família i seed, el que en torna `stability`; i `counters`, per família i
    `(seed, configuració)`, la part del comptador de dies de cada model ajustat de la
    família que en té. Un model sense cap contribució no hi entra.
    """
    rows = []

    def add(measure, family, value, unit=UNITLESS, **keys):
        rows.append(
            {"measure": measure, "family": family, **keys, "value": value, "unit": unit}
        )

    means = {}
    for family, fits in models.items():
        kept = {key: found for key, found in fits.items() if found}
        if not kept:
            continue
        first = next(iter(kept.values()))
        unit = first["unit"]
        for (seed, rank), found in kept.items():
            for name, value in found["shares"].items():
                add("model_share", family, value, seed=seed, rank=rank, subject=name)
        for (seed, rank), found in kept.items():
            for name, value in found["slopes"].items():
                add(
                    "model_slope",
                    family,
                    value,
                    f"{unit} per GW",
                    seed=seed,
                    rank=rank,
                    subject=name,
                )
        # Caraker et al. (2026b); un sol model no en fa cap parell.
        pairs = spearman_pairs(
            np.array([list(found["shares"].values()) for found in kept.values()])
        )
        add("spearman_mean", family, float(np.mean(pairs)) if len(pairs) else math.nan)
        orders = collections.Counter(_order(found["shares"]) for found in kept.values())
        for written, count in orders.most_common():
            add("group_order", family, count, subject=written)
        by_seed = stabilities[family]
        for name in first["shares"]:
            deviations = [found[name][0] for found in by_seed.values()]
            consensus = float(np.mean([found[name][1] for found in by_seed.values()]))
            present_deviations = [one for one in deviations if not np.isnan(one)]
            index = (
                float(np.mean(present_deviations)) / consensus
                if present_deviations
                else math.nan
            )
            add("stability_index", family, index, subject=name)
            add("consensus_importance", family, consensus, unit, subject=name)
        # Caraker et al. (2026a): el signe de la diferència de magnitud de cada model,
        # contra el de la mitjana.
        table = np.array(
            [list(found["magnitudes"].values()) for found in kept.values()]
        )
        gaps = table[:, :, None] - table[:, None, :]
        flips = (np.sign(gaps) != np.sign(gaps.mean(axis=0))).mean(axis=0)
        names = list(first["magnitudes"])
        for i, j in itertools.combinations(range(len(names)), 2):
            add(
                "flip_rate",
                family,
                float(flips[i, j]),
                subject=names[i],
                other_subject=names[j],
            )
        seeds = [seed for seed, _ in kept]
        for kind, measure in (
            ("shares", "between_seeds_share"),
            ("slopes", "between_seeds_slope"),
        ):
            for name in first[kind]:
                values = [found[kind][name] for found in kept.values()]
                add(measure, family, _between_seeds(values, seeds), subject=name)
        means[family] = {
            seed: {
                kind: np.mean(
                    [
                        list(found[kind].values())
                        for (one, _), found in kept.items()
                        if one == seed
                    ],
                    axis=0,
                )
                for kind in ("shares", "slopes")
            }
            for seed in dict.fromkeys(seeds)
        }
        if "own_hour" in first:
            by_hour = {
                kind: {
                    seed: np.mean(
                        [
                            found[kind]
                            for (one, _), found in kept.items()
                            if one == seed
                        ],
                        axis=0,
                    )
                    for seed in dict.fromkeys(seeds)
                }
                for kind in ("own_hour", "late_evening")
            }
            for kind, found in by_hour.items():
                for seed, values in found.items():
                    for hour, value in enumerate(values):
                        add(kind, family, float(value), unit, seed=seed, hour=hour)
                for hour, value in enumerate(np.mean(list(found.values()), axis=0)):
                    add(kind, family, float(value), unit, hour=hour)
    for family, other in itertools.combinations(means, 2):
        for seed in means[family].keys() & means[other].keys():
            found = _agreements(
                (means[family][seed]["shares"], means[family][seed]["slopes"]),
                (means[other][seed]["shares"], means[other][seed]["slopes"]),
            )
            for name, value in found.items():
                add(name, family, value, other_family=other, seed=seed)
    for family, shares in counters.items():
        for (seed, rank), value in shares.items():
            add("counter_share", family, value, seed=seed, rank=rank)
    return rows
