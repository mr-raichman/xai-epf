"""Les figures d'una passada, adaptades de Trebbien et al. (2023).

Es dibuixen del que la passada va desar, sense tornar a ajustar res.
"""

import importlib
import sys

import numpy as np

from epf import Refusal, paths
from epf.contract import figure
from epf.contract.explanation import PRICE_UNITS
from epf.contract.figure import plt
from epf.run import families, ljung_box, tables


def _performance(where, described, colours):
    """Dibuixa l'R² i el MAE de cada família a les setmanes de prova, una per fila.

    La millor configuració de cada seed amb una marca del color de la família, i les
    altres amb una de petita i transparent al darrere. La primera família, a dalt.
    """
    shown = [family for family in families.EVERY if family in described]
    panels, axes = plt.subplots(
        1, 2, figsize=figure.size(figure.rows_height(len(shown))), sharey=True
    )
    positions = np.arange(len(shown))[::-1]
    for axis, metric, label in (
        (axes[0], "r2", "R² a les setmanes de prova"),
        (axes[1], "mae", figure.unit_label("MAE a les setmanes de prova", PRICE_UNITS)),
    ):
        for position, family in zip(positions, shown, strict=True):
            for best, marker, opacity in (
                (False, figure.MARKER_SIZE / 2, figure.POINT_OPACITY),
                (True, figure.MARKER_SIZE, 1),
            ):
                found = [
                    float(model[metric])
                    for model in described[family]
                    if (int(model["rank"]) == 0) == best
                ]
                axis.plot(
                    found,
                    [position] * len(found),
                    "o",
                    markersize=marker,
                    color=colours[family],
                    alpha=opacity,
                )
        axis.set_xlabel(label)
    axes[0].set_yticks(positions, [figure.FAMILY_LABELS[family] for family in shown])
    figure.save(panels, where, "performance")


def draw(where):
    """Dibuixa les figures de les famílies completes d'una passada, `where`.

    Una família és completa quan cada seed del seu pla té els seus fitxers, i les
    figures de totes juntes i el test de Ljung-Box són de les completes; `scores.csv`,
    de cada model ajustat que hi ha. Refusa una passada sense cap família completa.
    """
    colours = figure.family_colours(families.MODELS)
    described = {}
    folders = [path for path in where.iterdir() if path.is_dir()]
    # En l'ordre de les famílies; una carpeta d'una família que no hi és fa fallar el
    # dibuix, en lloc de quedar-se sense figures sense dir-ho.
    for folder in sorted(folders, key=lambda path: families.EVERY.index(path.name)):
        if not families.complete(folder):
            continue
        models = figure.saved(folder)
        described[folder.name] = models
        module = families.MODULES[folder.name]["figures"]
        if module is not None:
            importlib.import_module(module).draw(folder, models, colours[folder.name])
    if not described:
        raise Refusal(f"la passada {where.name} no té cap família completa")
    _performance(where, described, colours)
    ljung_box.report(where, described, colours)
    tables.scores(where)


def main(arguments):
    """Dibuixa de nou les figures de la passada del seu tipus, la comanda `epf figures`.

    De mesura, o d'assaig amb `--smoke`, i de les seves famílies completes. Refusa que
    no n'hi hagi cap.
    """
    found = paths.named_runs(arguments.smoke)
    if not found:
        kind = "d'assaig" if arguments.smoke else "de mesura"
        runs = paths.relative(paths.runs(arguments.smoke))
        raise Refusal(f"no hi ha cap passada {kind} a {runs}/")
    where = found[-1]
    print(f"Passada {where.name}\n", file=sys.stderr)
    draw(where)
