"""Les figures d'una passada, adaptades de Trebbien et al. (2023).

Es dibuixen del que la passada va desar, sense tornar a ajustar res.
"""

import importlib
import sys

from epf import Refusal, paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.run import families, residuals, tables

# Cada família amb el `figures.py` del seu model, o `None` si no en té cap. Hi són
# totes: una carpeta d'una família que no hi és fa fallar el dibuix, en lloc de
# quedar-se sense figures sense dir-ho.
FIGURES = {
    "ddnn_jsu": "epf.models.nn.ddnn.figures",
    "ddnn_normal": "epf.models.nn.ddnn.figures",
    "dnn": None,
    "lightgbm": "epf.models.lightgbm.figures",
    "date_probe": "epf.models.lightgbm.figures",
    "elastic_net": "epf.models.elastic_net.figures",
    "floor": "epf.models.floor.figures",
    "naive_similar_day": None,
    "naive_previous_week": None,
    "naive_previous_day": None,
}

# L'alçada de la figura de rendiment, en mm.
PERFORMANCE_HEIGHT = 70


def _performance(where, described, colours):
    """Dibuixa l'R² i el MAE de cada família a les setmanes de prova.

    La millor configuració de cada seed amb una marca del color de la família, i les
    altres amb una de petita i transparent al darrere.
    """
    shown = [family for family in families.EVERY if family in described]
    panels, axes = plt.subplots(1, 2, figsize=figure.size(PERFORMANCE_HEIGHT))
    for axis, metric, label in (
        (axes[0], "r2", "R²"),
        (axes[1], "mae", "MAE, EUR/MWh"),
    ):
        for position, family in enumerate(shown):
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
                    [position] * len(found),
                    found,
                    "o",
                    markersize=marker,
                    color=colours[family],
                    alpha=opacity,
                )
        axis.set_xticks(range(len(shown)), shown, rotation=90)
        axis.set_ylabel(label)
        axis.set_title(f"{label} a les setmanes de prova, la millor de cada seed")
    figure.save(panels, where, "performance")


def draw(where):
    """Dibuixa totes les figures d'una passada, `where`, del que hi ha desat."""
    colours = figure.family_colours(families.MODELS)
    described = {}
    folders = [path for path in where.iterdir() if path.is_dir()]
    # En l'ordre de les famílies.
    for folder in sorted(folders, key=lambda path: list(FIGURES).index(path.name)):
        models = figure.saved(folder)
        if not models:
            continue
        described[folder.name] = models
        if FIGURES[folder.name] is not None:
            importlib.import_module(FIGURES[folder.name]).draw(
                folder, models, colours[folder.name]
            )
    if described:
        _performance(where, described, colours)
        residuals.report(where, described, colours)
        tables.scores(where)
        tables.date_probe(where)


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
