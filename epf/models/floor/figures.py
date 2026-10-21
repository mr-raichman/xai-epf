"""La figura del sòl de l'ordre de mèrit, del que la passada va desar."""

import numpy as np

from epf.contract import figure
from epf.contract.figure import plt

# L'alçada de la figura, en mm, i les caselles de la densitat a cada eix.
HEIGHT = 95
BINS = 80


def draw(folder, models, colour):
    """Dibuixa la figura del sòl a `folder`, dels seus models ajustats.

    El preu contra la càrrega residual a les files d'entrenament de la primera seed, en
    densitat, i el cúbic de cada seed, de `colour`, el color de la família, com la
    figura de Trebbien et al. (2023). Una casella sense cap hora queda en blanc.
    """
    first = models[0]
    load = first["load_training"] / figure.PER_GW
    panels, axis = plt.subplots(figsize=figure.size(HEIGHT))
    image = axis.hist2d(
        load,
        first["actual_training"],
        bins=BINS,
        cmin=1,
        rasterized=BINS**2 > figure.DENSE_POINTS,
    )[3]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), 200)
    for model in models:
        axis.plot(
            span,
            np.polyval(model["coefficients"], span * figure.PER_GW),
            color=colour,
        )
    axis.set_xlabel("càrrega residual prevista, GW")
    axis.set_ylabel("preu, EUR/MWh")
    axis.set_title(
        "El sòl de l'ordre de mèrit: les files d'entrenament de la "
        f"primera seed i el cúbic de cada una, seeds: {len(models)}"
    )
    figure.save(panels, folder, "floor")
