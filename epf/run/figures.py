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
from epf.run import families, residuals, tables


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
    """Dibuixa totes les figures d'una passada, `where`, del que hi ha desat."""
    colours = figure.family_colours(families.MODELS)
    described = {}
    folders = [path for path in where.iterdir() if path.is_dir()]
    # En l'ordre de les famílies; una carpeta d'una família que no hi és fa fallar el
    # dibuix, en lloc de quedar-se sense figures sense dir-ho.
    for folder in sorted(folders, key=lambda path: families.EVERY.index(path.name)):
        models = figure.saved(folder)
        if not models:
            continue
        described[folder.name] = models
        module = families.MODULES[folder.name]["figures"]
        if module is not None:
            importlib.import_module(module).draw(folder, models, colours[folder.name])
    if described:
        _performance(where, described, colours)
        residuals.report(where, described, colours)
        tables.scores(where)


def main(arguments):
    """Dibuixa de nou les figures d'una passada, la comanda `epf figures`.

    La passada és la que es nomeni o, sense nom, l'última acabada del seu tipus, de
    mesura si no és `--smoke`. Refusa un nom que no és cap passada i, sense nom, que no
    n'hi hagi cap d'acabada.
    """
    if arguments.run is None:
        found = [
            where for where in paths.runs(arguments.smoke) if families.finished(where)
        ]
        if not found:
            kind = "d'assaig" if arguments.smoke else "de mesura"
            raise Refusal(
                f"no hi ha cap passada {kind} acabada a {paths.relative(paths.RUNS)}/"
            )
        where = found[-1]
    else:
        # Només el nom d'una passada que hi és, mai un camí: el que s'escriu no tria on.
        found = {
            path.name: path for smoke in (False, True) for path in paths.runs(smoke)
        }
        if arguments.run not in found:
            runs = paths.relative(paths.RUNS)
            raise Refusal(f'no hi ha cap passada "{arguments.run}" a {runs}/')
        where = found[arguments.run]
    print(f"Passada {where.name}\n", file=sys.stderr)
    draw(where)
