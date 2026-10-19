"""Les figures de l'arbre i de la sonda de datació, del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt

# Socis de cada sèrie a les interaccions, els que més hi interactuen: la regla que fa de
# les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3


def _interactions(folder, model, present):
    """Dibuixa les interaccions de cada sèrie al millor model ajustat, `model`.

    És el de la primera llavor. Per sèrie de `present`, la dependència sencera, l'efecte
    sense interaccions i la interacció amb els `PARTNERS` socis més forts, amb el color
    del valor de cada soci.
    """
    columns = list(model["columns"])
    rows = model["interaction_rows"]
    inputs, interactions = model["inputs"][rows], model["interactions"]
    values = model["values"][rows]
    panels, axes = plt.subplots(
        len(present),
        2 + PARTNERS,
        figsize=(4 * (2 + PARTNERS), 3.5 * len(present)),
        facecolor=figure.SURFACE,
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
            (2 * interactions[:, j, k], inputs[:, k], f"amb {columns[k]}")
            for k in partners
        ]
        for axis, (y, colour, title) in zip(row, panels_of_row, strict=True):
            if colour is None:
                axis.plot(x, y, ".", color=figure.SERIES, markersize=2, alpha=0.5)
            else:
                image = axis.scatter(x, y, c=colour, cmap=figure.MAGNITUDE, s=3)
                figure.colorbar(panels, image, axis, "valor del soci")
            axis.set_xlabel(f"{figure.GROUP_LABELS[name]}, GW", color=figure.MUTED_INK)
            figure.style(axis, title)
    figure.save(panels, folder, "interactions")


def draw(folder, models):
    """Dibuixa les figures d'una família a `folder`, dels seus models ajustats.

    La importància, la dependència i, si el fitxer en porta, les interaccions.
    """
    figure.importance(folder, models)
    present = figure.present(models[0])
    if present:
        figure.dependence(folder, models, present)
        if "interactions" in models[0].files:
            _interactions(folder, models[0], present)
