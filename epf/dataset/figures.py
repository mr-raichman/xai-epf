"""La figura de cada valor horari de cada sèrie del dataset."""

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.information import INFORMATION
from epf.parameters import HOURS_PER_DAY

# L'alçada de la figura, en mm.
HEIGHT = 46


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

    title, unit = INFORMATION[name].title, figure.GROUP_UNITS[name]
    axis.set_title(f"{title}, de {frame.index[0].year} a {frame.index[-1].year}")
    axis.set_ylabel(figure.unit_label("valor horari", unit))
    axis.set_xlim(instants[0], instants[-1])
    figure.save(panels, paths.DATA_FIGURES / name, name)
