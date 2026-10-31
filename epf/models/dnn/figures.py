"""Les figures de la xarxa neuronal, del que la passada va desar."""

import numpy as np

from epf.contract import figure


def draw(folder, main, secondary, colour):
    """Dibuixa la dependència d'una família a `folder`, de `colour`, el seu color.

    `main` i `secondary` són els números de la Part II de la passada.
    """
    found = figure.first_explained(folder)
    if found is None:
        return
    seed, path = found
    with np.load(path) as model:
        figure.dependence(folder, seed, model, main, secondary, colour)
