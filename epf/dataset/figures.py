"""La figura de cada sèrie del conjunt de dades, cada valor horari, a
`data/figures/<sèrie>/<sèrie>.png`. Cada sèrie té la seva carpeta, on hi ha també la
figura dels seus valors allunyats dels veïns."""

import numpy as np
import pandas as pd

from epf import paths
from epf.contract import figure
from epf.contract.figure import GRID, INK, MUTED_INK, SERIES, SURFACE, plt
from epf.parameters import HOURS_PER_DAY

# El títol i la unitat de la figura de cada sèrie: el preu en EUR/MWh, les previsions
# D+1 en MW i el programa, l'energia programada de cada hora, en MWh.
LABELS = {
    "price": ("Preu horari del mercat diari espanyol", "EUR/MWh"),
    "demand_fcst": ("Previsió D+1 de la demanda", "MW"),
    "wind_fcst": ("Previsió D+1 de l'eòlica", "MW"),
    "solar_fcst": ("Previsió D+1 de la fotovoltaica", "MW"),
    "nuclear_sched": ("Generació nuclear programada (PBF)", "MWh"),
    "coal_gas_sched": ("Generació de carbó i gas programada (PBF)", "MWh"),
    "hydro_sched": (
        "Generació hidràulica programada, amb el turbinatge de bombeig (PBF)",
        "MWh",
    ),
}


def draw(name, frame):
    """Cada valor horari d'una sèrie, `frame` en dies per hores, a la carpeta de la
    sèrie. Una hora absent, com les de l'apagada, talla la línia en lloc d'unir-ne les
    veïnes."""
    instants = frame.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
        np.tile(np.arange(HOURS_PER_DAY), len(frame)), unit="h"
    )

    panels, axis = plt.subplots(figsize=(20, 6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.plot(instants, frame.to_numpy().ravel(), color=SERIES, linewidth=0.3)

    title, unit = LABELS[name]
    axis.set_title(
        f"{title}, {frame.index[0].year} a {frame.index[-1].year}",
        color=INK,
        loc="left",
    )
    axis.set_ylabel(unit, color=MUTED_INK)
    axis.set_xlim(instants[0], instants[-1])
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.tick_params(colors=MUTED_INK)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    figure.save(panels, paths.DATA_FIGURES / name, name)
