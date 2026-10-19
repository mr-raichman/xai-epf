"""Les figures d'una passada, adaptades de Trebbien et al. (2023).

Es dibuixen del que la passada va desar, sense tornar a ajustar res.
"""

import importlib
import sys

from epf import Refusal, paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.run import families, residuals

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


def _performance(where, described):
    """Dibuixa l'R² i el MAE de cada família a les setmanes de prova.

    La millor configuració de cada seed amb una marca, i les altres amb una de petita i
    grisa al darrere. Només les famílies explicades tenen color; els bancs de
    comparació, que no desen cap contribució, van en gris.
    """
    shown = [family for family in families.EVERY if family in described]
    explained = [family for family in shown if "values" in described[family][0].files]
    colour = {name: hue for hue, name in figure.colours(explained)}
    panels, axes = plt.subplots(
        1, 2, figsize=(max(8, 1.1 * len(shown)), 5), facecolor=figure.SURFACE
    )
    for axis, metric, label in (
        (axes[0], "r2", "R²"),
        (axes[1], "mae", "MAE, EUR/MWh"),
    ):
        for position, family in enumerate(shown):
            for best, size in ((False, 2), (True, 4)):
                found = [
                    float(model[metric])
                    for model in described[family]
                    if (int(model["rank"]) == 0) == best
                ]
                axis.plot(
                    [position] * len(found),
                    found,
                    "o",
                    markersize=size,
                    color=colour.get(family, figure.MUTED_INK)
                    if best
                    else figure.MUTED_INK,
                )
        axis.set_xticks(range(len(shown)), shown, fontsize=6, rotation=90)
        axis.set_ylabel(label, color=figure.MUTED_INK)
        figure.style(axis, f"{label} a les setmanes de prova, la millor de cada seed")
    figure.save(panels, where, "performance")


def draw(where):
    """Dibuixa totes les figures d'una passada, `where`, del que hi ha desat."""
    described = {}
    folders = [path for path in where.iterdir() if path.is_dir()]
    # En l'ordre de les famílies.
    for folder in sorted(folders, key=lambda path: list(FIGURES).index(path.name)):
        models = figure.saved(folder)
        if not models:
            continue
        described[folder.name] = models
        if FIGURES[folder.name] is not None:
            importlib.import_module(FIGURES[folder.name]).draw(folder, models)
    if described:
        _performance(where, described)
        residuals.report(where, described)


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
