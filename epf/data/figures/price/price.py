"""Gràfic de tots els preus horaris del mercat diari espanyol."""
import pandas as pd

from epf import paths
from epf.data.price import OMIE_FROM, load
from epf.figure import DPI, GRID, INK, MUTED_INK, SERIES, SURFACE, plt

OUTPUT = paths.PRICE_FIGURES / "price.png"


def main():
    price = load()
    hourly = price.stack()
    instants = hourly.index.get_level_values(0) + pd.to_timedelta(
        [int(hour[1:]) for hour in hourly.index.get_level_values(1)], unit="h")

    figure, axis = plt.subplots(figsize=(20, 6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.plot(instants, hourly.to_numpy(), color=SERIES, linewidth=0.3)

    seam = pd.Timestamp(OMIE_FROM)
    axis.axvline(seam, color=MUTED_INK, linewidth=1, linestyle="--")
    axis.annotate("ESIOS", (seam, 1), xycoords=("data", "axes fraction"), xytext=(-6, -14),
                  textcoords="offset points", ha="right", color=MUTED_INK)
    axis.annotate("OMIE", (seam, 1), xycoords=("data", "axes fraction"), xytext=(6, -14),
                  textcoords="offset points", ha="left", color=MUTED_INK)

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

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=DPI)
    print(f"desat a {OUTPUT}")
