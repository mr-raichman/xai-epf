"""Les mesures de la Part II, de les contribucions que desa cada model ajustat explicat.

Cada seed hi compta una vegada, la mitjana de les seves configuracions, totes igual.
"""

import itertools

import numpy as np

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


def slope(model, values, name):
    """Torna el pendent per GW de la contribució del grup `name` contra el seu valor."""
    x, y = (np.asarray(one, dtype=np.float64) for one in pairs(model, values, name))
    # En `float64`: amb els `float32` desats, `polyfit` ajusta en `float32`.
    return float(np.polyfit(x / PER_GW, y, 1)[0])


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
    `partners`.
    """
    values = contributions(model)
    if not np.any(values):
        return None
    _, shares = importances(values, model["groups"])
    found = {
        "shares": {name: float(shares[name]) for name in ordered(shares)},
        "slopes": {name: slope(model, values, name) for name in present(model)},
        "unit": str(model["unit"]),
    }
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
