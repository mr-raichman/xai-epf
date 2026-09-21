"""Gràfic de la puntuació de cada preu horari respecte dels seus veïns."""
import pathlib

import matplotlib
import numpy as np
import pandas as pd

from epf.config.market import HOURS_PER_DAY
from epf.data import prices
from epf.data.outliers import hampel_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

OUTPUT = pathlib.Path(__file__).with_suffix(".png")

# Veïns: les hores adjacents de la sèrie cronològica, i la mateixa hora en els
# dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# La majoria de puntuacions són inferiors a 10 i les més altes arriben als milers:
# eix lineal fins a 1 i logarítmic a partir d'aquí.
LINEAR_UP_TO = 1

SERIES_COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

frame = prices.load()
by_hours = hampel_score(frame.to_numpy().ravel(), HOUR_HALF_WIDTH).reshape(frame.shape)
by_days = np.column_stack([hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
                           for column in frame.columns])


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


figure, (top, bottom) = plt.subplots(2, 1, figsize=(20, 12), facecolor=SURFACE)

hours = np.arange(HOURS_PER_DAY)
top.add_collection(LineCollection([np.column_stack([hours, day]) for day in by_hours],
                                  colors=SERIES_COLOR, linewidths=0.3, alpha=0.3))
top.set_xlim(0, HOURS_PER_DAY - 1)
top.set_ylim(0, np.nanmax(by_hours) * 1.5)
top.set_xticks(hours)
top.set_xlabel("hora del dia", color=MUTED_INK)
style(top, f"Respecte de les hores adjacents (±{HOUR_HALF_WIDTH} h), un dia per línia",
      "distància a la mediana, en MAD")

instants = frame.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
    np.tile(hours, len(frame)), unit="h")
bottom.scatter(instants, by_days.ravel(), s=1, color=SERIES_COLOR, linewidths=0)
bottom.set_xlim(instants[0], instants[-1])
bottom.set_ylim(0, np.nanmax(by_days) * 1.5)
style(bottom, f"Respecte de la mateixa hora dels dies del voltant (±{DAY_HALF_WIDTH} dies)",
      "distància a la mediana, en MAD")

figure.suptitle(f"Distància de cada preu horari als seus veïns, {frame.index[0].year} a "
                f"{frame.index[-1].year}", color=INK, x=0.01, ha="left")
figure.tight_layout()
figure.savefig(OUTPUT, dpi=300)
print(f"desat a {OUTPUT}")
