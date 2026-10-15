"""La figura de tots els preus horaris del mercat diari espanyol, a `data/figures/price.png`."""
import pandas as pd

from epf import paths
from epf.contract import figure
from epf.contract.figure import GRID, INK, MUTED_INK, SERIES, SURFACE, plt


def draw(price):
    """Cada preu horari de `price`, en dies per hores."""
    hourly = price.stack()
    instants = hourly.index.get_level_values(0) + pd.to_timedelta(
        [int(hour[1:]) for hour in hourly.index.get_level_values(1)], unit="h")

    panels, axis = plt.subplots(figsize=(20, 6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.plot(instants, hourly.to_numpy(), color=SERIES, linewidth=0.3)
    axis.set_title(f"Preu horari del mercat diari espanyol, {price.index[0].year} a "
                   f"{price.index[-1].year}", color=INK, loc="left")
    axis.set_ylabel("EUR/MWh", color=MUTED_INK)
    axis.set_xlim(instants[0], instants[-1])
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.tick_params(colors=MUTED_INK)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    figure.save(panels, paths.DATA_FIGURES, "price")
