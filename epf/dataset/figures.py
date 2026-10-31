"""Les figures de cada sèrie del dataset: cada valor horari i els seus histogrames."""

import numpy as np
import pandas as pd

from epf import information, paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.parameters import HOURS_PER_DAY

# L'alçada de la figura, en mm.
HEIGHT = 46

# Les caselles de cada histograma; l'hora, comptada des de 0, que en té un de sol, la
# que comença a les 12:00; i l'alçada de cada panell entre la seva amplada.
HISTOGRAM_BINS = 80
HISTOGRAM_HOUR = 12
HISTOGRAM_ASPECT = 0.3


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


def histograms(name, frame):
    """Dibuixa l'histograma dels valors d'una sèrie, de totes les hores i de les 12:00.

    `frame` és la sèrie en dies per hores, tot el seu tram. Dos panells que comparteixen
    l'eix dels valors, a `histograms_<sèrie>.svg`, a la carpeta de la sèrie; una hora
    absent, com les de l'apagada, no hi compta.
    """
    shown = (
        ("totes les hores", frame.to_numpy().ravel()),
        (f"hora {HISTOGRAM_HOUR + 1}", frame.iloc[:, HISTOGRAM_HOUR].to_numpy()),
    )
    panels, axes = figure.panel_grid(len(shown), 1, HISTOGRAM_ASPECT, sharex=True)
    for axis, (heading, values) in zip(axes.flat, shown, strict=True):
        axis.hist(
            values[~np.isnan(values)], bins=HISTOGRAM_BINS, color=figure.OTHER_COLOUR
        )
        axis.set_title(heading)
    unit = information.KINDS[information.SERIES[name].kind].unit
    figure.label_once(panels, axes, figure.unit_label("valor horari", unit), "hores")
    figure.save(panels, paths.DATA_FIGURES / name, f"histograms_{name}")
