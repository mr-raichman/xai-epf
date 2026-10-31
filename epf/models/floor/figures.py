"""La figura del sòl de l'ordre de mèrit, del que la passada va desar."""

import numpy as np

from epf.contract import figure, measures
from epf.contract.explanation import PRICE_UNITS
from epf.contract.figure import plt

# L'alçada de la figura, en mm, i les caselles de la densitat a cada eix.
HEIGHT = 95
BINS = 80

# Punts per on passa el cúbic de cada seed.
CURVE_POINTS = 200


def draw(folder, models, colour):
    """Dibuixa la figura del sòl a `folder`, dels seus models ajustats.

    El preu contra la càrrega residual a les files d'entrenament de la primera seed, en
    densitat, i el cúbic de cada seed, de `colour`, el color de la família, com la
    figura de Trebbien et al. (2023). Una casella sense cap hora queda en blanc.
    """
    first = models[0]
    load = first["load_training"] / measures.PER_GW
    panels, axis = plt.subplots(figsize=figure.size(HEIGHT))
    image = axis.hist2d(
        load,
        first["actual_training"],
        bins=BINS,
        cmin=1,
        rasterized=BINS**2 > figure.DENSE_POINTS,
    )[3]
    figure.colorbar(panels, image, axis, "hores")
    span = np.linspace(load.min(), load.max(), CURVE_POINTS)
    for model in models:
        axis.plot(
            span,
            np.polyval(model["coefficients"], span * measures.PER_GW),
            color=colour,
        )
    axis.set_xlabel(figure.unit_label("càrrega residual prevista $L$", "GW"))
    axis.set_ylabel(figure.unit_label("preu $P$", PRICE_UNITS))
    axis.set_title(
        "El sòl de l'ordre de mèrit: el cúbic de cada seed\n"
        r"$\hat{P} = \beta_0 + \beta_1 L + \beta_2 L^2 + \beta_3 L^3$"
        ", sobre l'entrenament de la primera"
    )
    figure.save(panels, folder, "floor")
