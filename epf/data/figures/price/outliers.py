"""Gràfic de la puntuació de cada preu horari respecte dels seus veïns."""
import matplotlib
import numpy as np
import pandas as pd

from epf import paths
from epf.data.price import load
from epf.data.handle_data import (DAY_HALF_WIDTH, HIGHEST, HOUR_HALF_WIDTH, highest,
                                 report)
from epf.market import HOURS_PER_DAY
from epf.palette import DPI, GRID, INK, MUTED_INK, SERIES, SURFACE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

OUTPUT = paths.PRICE_FIGURES / "outliers.png"

# La majoria de puntuacions són inferiors a 10 i les més altes arriben als milers:
# eix lineal fins a 1 i logarítmic a partir d'aquí.
LINEAR_UP_TO = 1

# Vermell de les puntuacions marcades, que és d'aquesta figura i de cap altra.
HIGHLIGHT_COLOR = "#e34948"


def style(axis, title, ylabel):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=INK, loc="left")
    axis.set_ylabel(ylabel, color=MUTED_INK)
    axis.set_yscale("symlog", linthresh=LINEAR_UP_TO)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.tick_params(colors=MUTED_INK)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def main():
    price = load()
    # Les puntuacions surten de l'informe, que és qui les calcula i les descriu: la
    # taula impresa i la figura han de ser dels mateixos números.
    by_hours, by_days = report("price", price)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(20, 12), facecolor=SURFACE)

    hours = np.arange(HOURS_PER_DAY)
    top.add_collection(LineCollection([np.column_stack([hours, day]) for day in by_hours],
                                      colors=SERIES, linewidths=0.3, alpha=0.3))
    top_marks = highest(by_hours)
    top.scatter(top_marks % HOURS_PER_DAY, by_hours.ravel()[top_marks], s=12,
                color=HIGHLIGHT_COLOR, zorder=3)
    top.set_xlim(0, HOURS_PER_DAY - 1)
    top.set_ylim(0, np.nanmax(by_hours) * 1.5)
    top.set_xticks(hours)
    top.set_xlabel("hora del dia", color=MUTED_INK)
    style(top, f"Respecte de les hores adjacents (±{HOUR_HALF_WIDTH} h), un dia per línia, "
          f"en vermell les {HIGHEST} més altes",
          "distància a la mediana, en MAD")

    instants = price.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
        np.tile(hours, len(price)), unit="h")
    bottom.scatter(instants, by_days.ravel(), s=1, color=SERIES, linewidths=0)
    bottom_marks = highest(by_days)
    bottom.scatter(instants[bottom_marks], by_days.ravel()[bottom_marks], s=12,
                   color=HIGHLIGHT_COLOR, zorder=3)
    bottom.set_xlim(instants[0], instants[-1])
    bottom.set_ylim(0, np.nanmax(by_days) * 1.5)
    style(bottom, f"Respecte de la mateixa hora dels dies del voltant (±{DAY_HALF_WIDTH} dies), "
          f"en vermell les {HIGHEST} més altes",
          "distància a la mediana, en MAD")

    figure.suptitle(f"Distància de cada preu horari als seus veïns, {price.index[0].year} a "
                    f"{price.index[-1].year}", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=DPI)
    print(f"desat a {OUTPUT}")
