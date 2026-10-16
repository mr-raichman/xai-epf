"""La figura del sòl de l'ordre de mèrit, dibuixada del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt


def draw(folder, seeds):
    """El preu contra la càrrega residual a les files d'entrenament de la primera
    llavor, en densitat, i el polinomi cúbic de cada llavor, la figura del sòl de
    l'article."""
    first = seeds[0]
    load = first["load_training"] / figure.PER_GW
    panels, axis = plt.subplots(figsize=(8, 5), facecolor=figure.SURFACE)
    image = axis.hist2d(load, first["actual_training"], bins=80, cmap=figure.MAGNITUDE)[
        3
    ]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), 200)
    for seed in seeds:
        axis.plot(
            span,
            np.polyval(seed["coefficients"], span * figure.PER_GW),
            color=figure.INK,
            linewidth=0.6,
        )
    axis.set_xlabel("càrrega residual prevista, GW", color=figure.MUTED_INK)
    axis.set_ylabel("preu, EUR/MWh", color=figure.MUTED_INK)
    figure.style(
        axis,
        "El sòl de l'ordre de mèrit: les files d'entrenament de la "
        f"primera llavor i el cúbic de cada una, llavors: {len(seeds)}",
    )
    figure.save(panels, folder, "floor")
