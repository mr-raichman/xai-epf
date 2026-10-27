"""La figura de cada valor horari de cada sèrie del dataset."""

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import figure
from epf.contract.figure import plt
from epf import information
from epf.parameters import HOURS_PER_DAY

# L'alçada de la figura, en mm.
HEIGHT = 46


def title(name):
    """Torna com es diu una sèrie a les figures del dataset.

    Com a les dels models, amb la primera lletra en majúscula.
    """
    label = information.SERIES[name].label
    return label[0].upper() + label[1:]


def draw(name, frame):
    """Dibuixa cada valor horari d'una sèrie, `frame` en dies per hores.

    La figura va a la carpeta de la sèrie, `<sèrie>.svg`, i una hora absent, com les de
    l'apagada, hi talla la línia en lloc d'unir-ne les veïnes.
    """
    instants = frame.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
        np.tile(np.arange(HOURS_PER_DAY), len(frame)), unit="h"
    )

    panels, axis = plt.subplots(figsize=figure.size(HEIGHT))
    values = frame.to_numpy().ravel()
    axis.plot(
        instants,
        values,
        color=figure.OTHER_COLOUR,
        **figure.line_style(values.size, overlapping=False),
    )

    unit = information.KINDS[information.SERIES[name].kind].unit
    axis.set_title(f"{title(name)}, de {frame.index[0].year} a {frame.index[-1].year}")
    axis.set_ylabel(figure.unit_label("valor horari", unit))
    axis.set_xlim(instants[0], instants[-1])
    figure.save(panels, paths.DATA_FIGURES / name, name)
