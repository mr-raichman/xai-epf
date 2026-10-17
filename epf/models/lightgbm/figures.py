"""Les figures de l'arbre i de la sonda de datació, dibuixades del que la passada va
desar: la importància de cada columna i de cada grup, la dependència de cada sèrie i les
seves interaccions."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt

# Socis de cada sèrie a les interaccions: els tres que més hi interactuen, la regla que
# fa de les "three selected features" de Trebbien et al. (2023).
PARTNERS = 3


def _interactions(folder, seed, present):
    """Al millor model de la primera llavor, per a cada sèrie: la dependència sencera,
    l'efecte sense interaccions i la interacció amb els tres socis més forts, amb el
    color del valor de cada soci."""
    columns = list(seed["columns"])
    rows = seed["interaction_rows"]
    inputs, interactions = seed["inputs"][rows], seed["interactions"]
    values = seed["values"][rows]
    panels, axes = plt.subplots(
        len(present),
        2 + PARTNERS,
        figsize=(4 * (2 + PARTNERS), 3.5 * len(present)),
        facecolor=figure.SURFACE,
        squeeze=False,
    )
    for row, name in zip(axes, present, strict=True):
        (j,) = figure.members(seed, name)
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


def draw(folder, seeds):
    figure.importance(folder, seeds)
    present = figure.present(seeds[0])
    if present:
        figure.dependence(folder, seeds, present)
        if "interactions" in seeds[0].files:
            _interactions(folder, seeds[0], present)
