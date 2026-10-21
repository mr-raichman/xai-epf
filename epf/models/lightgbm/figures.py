"""Les figures de l'arbre i de la sonda de datació, del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.explanation import PRICE_UNITS
from epf.contract.figure import plt

# Socis de cada sèrie a les interaccions, els que més hi interactuen: la regla que fa de
# les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3

# Panells de les interaccions per fila, i l'alçada de cada un entre la seva amplada.
PANELS_PER_ROW = 2
PANEL_ASPECT = 3 / 4


def _interactions(folder, model, present, colour):
    """Dibuixa les interaccions de cada sèrie al millor model ajustat, `model`.

    És el de la primera seed. Per sèrie de `present`, una figura,
    `interactions_<sèrie>.svg`: la dependència sencera i l'efecte sense interaccions, de
    `colour`, el color de la família, i la interacció amb els `PARTNERS` socis més
    forts, amb el color del valor de cada soci a l'escala contínua.
    """
    columns = list(model["columns"])
    rows = model["interaction_rows"]
    inputs, interactions = model["inputs"][rows], model["interactions"]
    values = model["values"][rows]
    count = 2 + PARTNERS
    lines = -(-count // PANELS_PER_ROW)
    for name in present:
        panels, axes = plt.subplots(
            lines,
            PANELS_PER_ROW,
            figsize=figure.size(lines * figure.WIDTH / PANELS_PER_ROW * PANEL_ASPECT),
            squeeze=False,
        )
        for axis in axes.flat[count:]:
            axis.set_visible(False)
        (j,) = figure.members(model, name)
        x = inputs[:, j] / figure.PER_GW
        strength = np.abs(interactions[:, j, :]).mean(axis=0)
        strength[j] = -np.inf
        partners = np.argsort(strength)[-PARTNERS:][::-1]
        shown = [
            (values[:, j], None, "sencera"),
            (interactions[:, j, j], None, "sense interaccions"),
        ]
        shown += [
            (
                2 * interactions[:, j, k],
                k,
                f"amb {figure.column_label(columns[k], model['groups'][k])}",
            )
            for k in partners
        ]
        for axis, (y, k, title) in zip(axes.flat, shown, strict=False):
            if k is None:
                axis.scatter(x, y, color=colour, **figure.point_style(x.size))
            else:
                image = axis.scatter(
                    x, y, c=inputs[:, k], **figure.point_style(x.size, coloured=True)
                )
                unit = figure.GROUP_UNITS[model["groups"][k]]
                figure.colorbar(
                    panels, image, axis, figure.unit_label("valor del soci", unit)
                )
            axis.set_title(title)
            axis.set_xlabel(figure.unit_label(figure.GROUP_LABELS[name], "GW"))
        panels.supylabel(
            figure.unit_label("contribució", PRICE_UNITS),
            fontsize=figure.AXIS_TEXT_SIZE,
        )
        figure.save(panels, folder, f"interactions_{name}")


def draw(folder, models, colour):
    """Dibuixa les figures d'una família a `folder`, dels seus models ajustats.

    La importància, la dependència i, si el fitxer en porta, les interaccions, de
    `colour`, el color de la família.
    """
    figure.importance(folder, models, colour)
    present = figure.present(models[0])
    if present:
        figure.dependence(folder, models, present, colour)
        if "interactions" in models[0].files:
            _interactions(folder, models[0], present, colour)
