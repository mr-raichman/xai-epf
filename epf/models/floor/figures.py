"""La figura del sòl de l'ordre de mèrit, del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt


def draw(folder, models):
    """Dibuixa la figura del sòl a `folder`, dels seus models ajustats.

    El preu contra la càrrega residual a les files d'entrenament de la primera seed, en
    densitat, i el cúbic de cada seed, com la figura de Trebbien et al. (2023).
    """
    first = models[0]
    load = first["load_training"] / figure.PER_GW
    panels, axis = plt.subplots(figsize=(8, 5), facecolor=figure.SURFACE)
    image = axis.hist2d(load, first["actual_training"], bins=80, cmap=figure.MAGNITUDE)[
        3
    ]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), 200)
    for model in models:
        axis.plot(
            span,
            np.polyval(model["coefficients"], span * figure.PER_GW),
            color=figure.INK,
            linewidth=0.6,
        )
    axis.set_xlabel("càrrega residual prevista, GW", color=figure.MUTED_INK)
    axis.set_ylabel("preu, EUR/MWh", color=figure.MUTED_INK)
    figure.style(
        axis,
        "El sòl de l'ordre de mèrit: les files d'entrenament de la "
        f"primera seed i el cúbic de cada una, seeds: {len(models)}",
    )
    figure.save(panels, folder, "floor")
