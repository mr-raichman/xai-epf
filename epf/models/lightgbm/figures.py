"""Les figures de l'arbre i de la sonda de datació, del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt

# Socis de cada sèrie a les interaccions, els que més hi interactuen: la regla que fa de
# les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3

# L'alçada de cada panell de les interaccions entre la seva amplada.
PANEL_ASPECT = 3.5 / 4


def _interactions(folder, model, present, colour):
    """Dibuixa les interaccions de cada sèrie al millor model ajustat, `model`.

    És el de la primera seed. Per sèrie de `present`, la dependència sencera i l'efecte
    sense interaccions, de `colour`, el color de la família, i la interacció amb els
    `PARTNERS` socis més forts, amb el color del valor de cada soci a l'escala contínua.
    """
    columns = list(model["columns"])
    rows = model["interaction_rows"]
    inputs, interactions = model["inputs"][rows], model["interactions"]
    values = model["values"][rows]
    across = 2 + PARTNERS
    panels, axes = plt.subplots(
        len(present),
        across,
        figsize=figure.size(len(present) * figure.WIDTH / across * PANEL_ASPECT),
        squeeze=False,
    )
    for row, name in zip(axes, present, strict=True):
        (j,) = figure.members(model, name)
        x = inputs[:, j] / figure.PER_GW
        strength = np.abs(interactions[:, j, :]).mean(axis=0)
        strength[j] = -np.inf
        partners = np.argsort(strength)[-PARTNERS:][::-1]
        panels_of_row = [
            (values[:, j], None, "sencera"),
            (interactions[:, j, j], None, "sense interaccions"),
        ]
        panels_of_row += [
            (
                2 * interactions[:, j, k],
                inputs[:, k],
                f"amb {figure.column_label(columns[k], model['groups'][k])}",
            )
            for k in partners
        ]
        for axis, (y, partner, title) in zip(row, panels_of_row, strict=True):
            if partner is None:
                axis.scatter(x, y, color=colour, **figure.point_style(x.size))
            else:
                image = axis.scatter(x, y, c=partner, **figure.point_style(x.size))
                figure.colorbar(panels, image, axis, "valor del soci")
            axis.set_title(title)
        row[0].set_xlabel(f"{figure.GROUP_LABELS[name]}, GW")
    figure.save(panels, folder, "interactions")


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
