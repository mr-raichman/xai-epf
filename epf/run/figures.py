"""Les figures d'una passada, adaptades de Trebbien et al. (2023).

Es dibuixen del que la passada va desar, sense tornar a ajustar res, i els números que
hi surten són els dels seus fitxers.
"""

import importlib
import json
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from epf import Refusal, information, parameters, paths
from epf.contract import figure, measures
from epf.contract.explanation import PRICE_UNITS
from epf.contract.figure import plt
from epf.parameters import HOURS_PER_DAY
from epf.run import families, ljung_box, tables

# Els panells per fila de l'rMAE per franges, i l'alçada de cada un entre la seva
# amplada.
BREAKDOWN_PER_ROW = 3
BREAKDOWN_ASPECT = 0.8

# Cada franja de l'rMAE, pel nom de la seva figura: la mesura, la clau on va la franja i
# el títol de l'eix.
BREAKDOWNS = {
    "rmae_by_hour": ("rmae", "hour", "hora del dia"),
    "rmae_by_year": ("rmae_year", "subject", "any"),
    "rmae_by_decile": (
        "rmae_decile",
        "subject",
        figure.unit_label("límit superior del decil del preu real", PRICE_UNITS),
    ),
}

# Cada quantes hores s'escriu una hora a l'eix.
HOUR_STEP = 3

# Els caràcters que poden tenir en total les xifres d'un eix de franges per anar
# planes; amb més, van dretes.
LEVEL_CHARACTERS = 20

# Els panells per fila de la densitat del preu real contra la previsió, i les seves
# caselles a cada eix.
DENSITY_PER_ROW = 4
DENSITY_BINS = 80

# L'alçada d'un panell d'una família, de l'amplada de la figura, entre la seva amplada:
# la importància per grups i el vespre contra l'hora mateixa.
FAMILY_ASPECT = 0.32

# El beeswarm per grups: l'alçada d'un panell entre la seva amplada, la dispersió dels
# punts al voltant de la fila d'un grup, el diàmetre d'un punt en punts, a ull, entre
# els de 0.55 i 2 que s'hi van provar, i els quantils del valor del grup entre els quals
# va l'escala de colors.
BEESWARM_ASPECT = 0.3
BEESWARM_JITTER = 0.3
BEESWARM_DOT = 1.4
BEESWARM_RANGE = (0.05, 0.95)

# L'alçada de la correlació de Spearman, en mm.
SPEARMAN_HEIGHT = 70


def _complete(where):
    """Torna les famílies completes d'una passada, en l'ordre de les famílies.

    La carpeta d'una família que el codi ja no té, d'una passada començada amb un altre
    codi, en queda fora.
    """
    return [
        family
        for family in families.EVERY
        if (where / family).is_dir() and families.complete(where / family)
    ]


def _performance(where, main, complete, colours):
    """Dibuixa l'R², el MAE i l'rMAE de cada família a la prova, una per fila.

    De les famílies completes, `complete`, dels números de `part_i_main.csv`, `main`.
    La millor configuració de cada seed amb una marca del color de la família, i les
    altres amb una de petita i transparent al darrere; l'rMAE, amb una línia a 1, només
    si la passada té la còpia que el divideix. La primera família, a dalt.
    """
    # Tres títols llargs no hi caben l'un al costat de l'altre.
    metrics = [("r2", "R²"), ("mae", figure.unit_label("MAE", PRICE_UNITS))]
    if (main["measure"] == "rmae").any():
        metrics.append(("rmae", "rMAE"))
    panels, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figure.size(figure.rows_height(len(complete))),
        sharey=True,
        squeeze=False,
    )
    positions = np.arange(len(complete))[::-1]
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        for position, family in zip(positions, complete, strict=True):
            found = figure.rows(main, metric, family)
            for best, marker, opacity in (
                (False, figure.MARKER_SIZE / 2, figure.POINT_OPACITY),
                (True, figure.MARKER_SIZE, 1),
            ):
                values = found.loc[(found["rank"] == 0) == best, "value"].astype(float)
                axis.plot(
                    values,
                    [position] * len(values),
                    "o",
                    markersize=marker,
                    color=colours[family],
                    alpha=opacity,
                )
        if metric == "rmae":
            axis.axvline(1, **figure.REFERENCE_LINE)
        axis.set_xlabel(label)
    axes[0, 0].set_yticks(
        positions, [figure.FAMILY_LABELS[family] for family in complete]
    )
    figure.save(panels, where, "performance")


def _breakdowns(where, secondary, complete, colours):
    """Dibuixa l'rMAE de cada família per hora, per any i per decil del preu.

    Dels números de `part_i_secondary.csv`, `secondary`, de les famílies completes,
    `complete`: un panell per família, cada seed clara i la seva mitjana sòlida, i un
    darrer amb la mitjana de cada família; una línia a 1, el de la còpia que el
    divideix. Les hores, d'1 a 24, i els decils, pel seu límit de dalt. Sense la còpia,
    no en dibuixa cap.
    """
    for name, (measure, key, xlabel) in BREAKDOWNS.items():
        found = {
            family: figure.rows(secondary, measure, family).dropna(subset=[key])
            for family in complete
        }
        shown = [family for family in complete if not found[family].empty]
        if not shown:
            return
        panels, axes = figure.panel_grid(
            len(shown) + 1, BREAKDOWN_PER_ROW, BREAKDOWN_ASPECT, sharex=True
        )
        total = axes.flat[len(shown)]
        for axis, family in zip(axes.flat, shown, strict=False):
            rows = found[family].assign(place=found[family][key].astype(int))
            for seed, one in rows.groupby("seed", dropna=False):
                one = one.sort_values("place")
                x = one["place"] + (key == "hour")
                values = one["value"].astype(float)
                if pd.isna(seed):
                    axis.plot(x, values, "o-", color=colours[family])
                    total.plot(x, values, "o-", color=colours[family])
                else:
                    axis.plot(
                        x,
                        values,
                        "o-",
                        color=colours[family],
                        alpha=figure.BAND_OPACITY,
                    )
            axis.set_title(figure.FAMILY_LABELS[family])
        total.set_title("totes les famílies")
        places = sorted(set(found[shown[0]][key].astype(int)))
        if key == "hour":
            ticks = [place + 1 for place in places][::HOUR_STEP]
            labels = [str(tick) for tick in ticks]
        elif measure == "rmae_decile":
            ticks = places
            tops = [
                figure.number(secondary, "price_decile_top", None, subject=str(place))
                for place in places
            ]
            labels = [figure.catalan(f"{top:.0f}") for top in tops]
        else:
            ticks, labels = places, [str(place) for place in places]
        turned = 90 if sum(map(len, labels)) > LEVEL_CHARACTERS else 0
        for axis in axes.flat[: len(shown) + 1]:
            axis.axhline(1, **figure.REFERENCE_LINE)
            axis.set_xticks(ticks, labels, rotation=turned)
        figure.label_once(panels, axes, xlabel, "rMAE")
        figure.save(panels, where, name)


def _density(where, complete):
    """Dibuixa el preu real contra la previsió de cada família, en densitat.

    Totes les files de prova de la millor configuració de cada seed de les famílies
    completes, `complete`, un panell per família amb les mateixes caselles i la
    mateixa escala de colors, de manera que una sola barra val per a tots; la
    diagonal, on la previsió encerta.
    """
    best = tables.best_fits(where, complete)
    pairs = {
        family: [
            np.concatenate([fit[kind].ravel() for fit in fits.values()])
            for kind in ("actual", "predicted")
        ]
        for family, fits in best.items()
    }
    every = np.concatenate([np.concatenate(found) for found in pairs.values()])
    edges = np.linspace(every.min(), every.max(), DENSITY_BINS + 1)
    top = max(np.histogram2d(*found, bins=edges)[0].max() for found in pairs.values())
    panels, axes = figure.panel_grid(
        len(complete), DENSITY_PER_ROW, 1, sharex=True, sharey=True
    )
    for axis, family in zip(axes.flat, complete, strict=False):
        image = axis.hist2d(
            *pairs[family], bins=edges, cmin=1, vmin=1, vmax=top, rasterized=True
        )[3]
        axis.plot(edges[[0, -1]], edges[[0, -1]], **figure.REFERENCE_LINE)
        axis.set_title(figure.FAMILY_LABELS[family])
    figure.label_once(
        panels,
        axes,
        figure.unit_label("preu real", PRICE_UNITS),
        figure.unit_label("previsió", PRICE_UNITS),
    )
    figure.colorbar(panels, image, axes, "hores a la casella")
    figure.save(panels, where, "actual_against_forecast")


def _groups(where, main, secondary, interpreted, colours):
    """Dibuixa la importància de cada grup, una família interpretada per panell.

    Dels números de la Part II, `main` i `secondary`: la mitjana de les seeds en una
    barra, del color de la família, i cada model ajustat en un punt.
    """
    panels, axes = figure.panel_grid(
        len(interpreted), 1, FAMILY_ASPECT, sharex=True, sharey=True
    )
    for axis, family in zip(axes.flat, interpreted, strict=True):
        means = figure.rows(main, "share_mean", family, seed=None)
        names = list(means["subject"])
        positions = np.arange(len(names))
        axis.bar(
            positions,
            means["value"].astype(float),
            color=colours[family],
            width=figure.BAR_WIDTH,
        )
        models = figure.rows(secondary, "model_share", family)
        for _, one in models.groupby(["seed", "rank"]):
            axis.plot(
                positions,
                one.set_index("subject").loc[names, "value"].astype(float),
                "o",
                color=figure.TEXT_COLOUR,
                alpha=figure.POINT_OPACITY,
            )
        axis.set_title(figure.FAMILY_LABELS[family])
    axes.flat[len(interpreted) - 1].set_xticks(
        positions, [figure.GROUP_LABELS[name] for name in names], rotation=90
    )
    figure.label_once(panels, axes, ylabel="part de la importància")
    figure.save(panels, where, "groups")


def _beeswarm(where, interpreted):
    """Dibuixa la contribució de cada grup a cada fila, una família per panell.

    Del primer model ajustat de cada família amb alguna contribució, la millor
    configuració de la primera seed: un punt per fila, pintat pel valor del grup a
    l'hora de la fila entre els quantils de `BEESWARM_RANGE`, i el grup que més pesa, a
    dalt, com el resum per grups de Pesenti i O'Sullivan (2026).
    """
    panels, axes = figure.panel_grid(len(interpreted), 1, BEESWARM_ASPECT)
    rng = np.random.default_rng(parameters.seeded("beeswarm"))
    for axis, family in zip(axes.flat, interpreted, strict=True):
        _, path = figure.first_explained(where / family)
        with np.load(path) as model:
            values = np.asarray(measures.contributions(model), dtype=np.float64)
            groups = np.array([str(group) for group in model["groups"]])
            names = measures.ordered(dict.fromkeys(groups))
            by_group = np.stack(
                [values[:, groups == name].sum(axis=1) for name in names], axis=1
            )
            coloured = figure.group_values(model, len(values))
            unit = str(model["unit"])
        order = np.argsort(np.abs(by_group).mean(axis=0), kind="stable")
        for place, number in enumerate(order):
            value = coloured[names[number]]
            low, high = np.nanquantile(value, BEESWARM_RANGE)
            axis.scatter(
                by_group[:, number],
                place + rng.uniform(-BEESWARM_JITTER, BEESWARM_JITTER, len(value)),
                c=np.clip((value - low) / ((high - low) or 1), 0, 1),
                cmap=figure.VALUE_SCALE,
                s=BEESWARM_DOT**2,
                linewidths=0,
                rasterized=True,
            )
        axis.set_yticks(
            np.arange(len(order)), [figure.GROUP_LABELS[names[j]] for j in order]
        )
        axis.axvline(0, **figure.REFERENCE_LINE)
        axis.set_title(f"{figure.FAMILY_LABELS[family]}, en {unit}")
    figure.label_once(panels, axes, xlabel="contribució del grup")
    figure.save(panels, where, "beeswarm")


def _spearman(where, secondary, interpreted, colours):
    """Dibuixa la correlació de Spearman de cada parell de models, en violins.

    Dels números de la Part II, `secondary`: un violí per família interpretada, de les
    parts dels grups de cada model ajustat, i la seva mitjana en una línia. Unes
    correlacions totes iguals no fan cap violí, només la línia.
    """
    pairs = {}
    for family in interpreted:
        models = figure.rows(secondary, "model_share", family)
        names = list(dict.fromkeys(models["subject"]))
        table = models.pivot(index=["seed", "rank"], columns="subject", values="value")[
            names
        ]
        pairs[family] = measures.spearman_pairs(table.to_numpy(dtype=float))
    panels, axis = plt.subplots(figsize=figure.size(SPEARMAN_HEIGHT))
    positions = np.arange(len(interpreted))
    spread = [
        number
        for number, family in enumerate(interpreted)
        if len(pairs[family]) > 1 and np.ptp(pairs[family]) > 0
    ]
    if spread:
        drawn = axis.violinplot(
            [pairs[interpreted[number]] for number in spread],
            positions[spread],
            widths=figure.BAR_WIDTH,
            showextrema=False,
        )
        for body, number in zip(drawn["bodies"], spread, strict=True):
            body.set_facecolor(colours[interpreted[number]])
            body.set_edgecolor("none")
            body.set_alpha(1)
    half = figure.BAR_WIDTH / 2
    for position, family in zip(positions, interpreted, strict=True):
        mean = figure.number(secondary, "spearman_mean", family)
        axis.hlines(mean, position - half, position + half, color=figure.TEXT_COLOUR)
    axis.set_xticks(positions, [figure.FAMILY_LABELS[family] for family in interpreted])
    axis.set_ylabel("correlació de Spearman d'un parell de models")
    figure.save(panels, where, "spearman")


def _late_against_own(where, secondary, interpreted):
    """Dibuixa, per hora prevista, el vespre del dia abans contra la seva mateixa hora.

    Dels números de la Part II, `secondary`: de cada família interpretada que llegeix el
    preu del dia abans, la contribució absoluta mitjana del preu a la mateixa hora i la
    de les hores 22 a 24, la mitjana de les seeds, en dos tons del color del model; la
    llegenda, en grisos, perquè travessa famílies. Sense el preu del dia abans, no la
    dibuixa.
    """
    shown = [
        family
        for family in interpreted
        if not figure.rows(secondary, "own_hour", family, seed=None).empty
    ]
    if not shown:
        return
    shades = {"own_hour": 0, "late_evening": 2}
    panels, axes = figure.panel_grid(len(shown), 1, FAMILY_ASPECT, sharex=True)
    for axis, family in zip(axes.flat, shown, strict=True):
        model = families.MODULES[family]["model"]
        for kind, number in shades.items():
            found = figure.rows(secondary, kind, family, seed=None).sort_values("hour")
            axis.plot(
                found["hour"].astype(int) + 1,
                found["value"].astype(float),
                color=figure.shade(model, number),
            )
        unit = figure.rows(secondary, "own_hour", family, seed=None)["unit"].iloc[0]
        axis.set_title(f"{figure.FAMILY_LABELS[family]}, en {unit}")
        axis.set_xticks(np.arange(1, HOURS_PER_DAY + 1)[::HOUR_STEP])
    figure.label_once(panels, axes, "hora prevista", "contribució absoluta mitjana")
    panels.legend(
        handles=[
            Patch(facecolor=figure.grey(shades[kind]), label=label)
            for kind, label in (
                ("own_hour", "la mateixa hora del dia abans"),
                ("late_evening", "les hores 22 a 24 del dia abans, la mitjana"),
            )
        ],
        loc="outside upper center",
        ncols=2,
    )
    figure.save(panels, where, "late_against_own")


def _drawn(where, numbers, complete, colours):
    """Dibuixa les figures d'una passada dels seus números, `numbers`, i dels fitxers.

    Les de les famílies completes, `complete`, del color de cada una de `colours`.
    """
    explained = numbers[tables.PART_II_MAIN], numbers[tables.PART_II_SECONDARY]
    interpreted = [
        family
        for family in complete
        if json.loads(
            (where / family / paths.FAMILY_RECORD).read_text(encoding="utf-8")
        ).get("interpreted")
    ]
    for family in complete:
        module = families.MODULES[family]["figures"]
        if module is not None:
            importlib.import_module(module).draw(
                where / family, *explained, colours[family]
            )
    _performance(where, numbers[tables.PART_I_MAIN], complete, colours)
    _breakdowns(where, numbers[tables.PART_I_SECONDARY], complete, colours)
    _density(where, complete)
    ljung_box.report(where, numbers, colours)
    if interpreted:
        _groups(where, *explained, interpreted, colours)
        _beeswarm(where, interpreted)
        _spearman(where, explained[1], interpreted, colours)
        _late_against_own(where, explained[1], interpreted)


def draw(where):
    """Dibuixa les figures de les famílies completes d'una passada, `where`.

    Una família és completa quan cada seed del seu pla té els seus fitxers; abans,
    escriu els números de la passada, com diu `tables.write`. Refusa una passada sense
    cap família completa.
    """
    complete = _complete(where)
    if not complete:
        raise Refusal(f"la passada {where.name} no té cap família completa")
    numbers = tables.write(where)
    _drawn(where, numbers, complete, figure.family_colours(families.MODELS))


def main(arguments):
    """Dibuixa de nou les figures de la passada del seu tipus, la comanda `epf figures`.

    De mesura, o d'assaig amb `--smoke`, de l'information set `--set`, i de les
    seves famílies completes. Refusa un information set que no existeix i que no hi
    hagi cap passada.
    """
    set_name = information.resolved(arguments.set_name).name
    found = paths.named_runs(arguments.smoke, set_name)
    if not found:
        kind = "d'assaig" if arguments.smoke else "de mesura"
        runs = paths.relative(paths.runs(arguments.smoke, set_name))
        raise Refusal(f"no hi ha cap passada {kind} a {runs}/")
    where = found[-1]
    print(f"Passada {where.name}\n", file=sys.stderr)
    draw(where)
