"""Gràfic de la puntuació de cada preu horari respecte dels seus veïns."""
import matplotlib
import numpy as np
import pandas as pd

from epf import paths
from epf.data import prices
from epf.data.outliers import hampel_score, window_median_mad
from epf.market import HOURS_PER_DAY
from epf.palette import DPI, GRID, INK, MUTED_INK, SERIES, SURFACE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

OUTPUT = paths.PLOTS / "outliers.png"

# Veïns: les hores adjacents de la sèrie cronològica, i la mateixa hora en els
# dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# La majoria de puntuacions són inferiors a 10 i les més altes arriben als milers:
# eix lineal fins a 1 i logarítmic a partir d'aquí.
LINEAR_UP_TO = 1

# Puntuacions més altes de cada panell, marcades en vermell.
HIGHEST = 100

# Vermell de les puntuacions marcades, que és d'aquesta figura i de cap altra.
HIGHLIGHT_COLOR = "#e34948"


def highest(scores):
    """Posicions de les `HIGHEST` puntuacions més altes, sense comptar les NaN."""
    flat = np.nan_to_num(scores.ravel(), nan=-np.inf)
    return np.argsort(flat)[-HIGHEST:]


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


def describe(days, values, name, scores, median, mad):
    """Resum de les puntuacions més altes: com és la finestra i on cauen."""
    marks = highest(scores)
    years = pd.Series(days.repeat(HOURS_PER_DAY)[marks].year).value_counts().sort_index()
    print(f"{name}, les {HIGHEST} puntuacions més altes:")
    print(f"  MAD de la finestra, mediana {np.median(mad[marks]):.2f} EUR/MWh, "
          f"màxim {mad[marks].max():.2f}")
    print(f"  mediana de la finestra, mediana {np.median(median[marks]):.2f} EUR/MWh")
    print(f"  distància del preu a la mediana, mediana "
          f"{np.median(np.abs(values - median)[marks]):.2f} EUR/MWh")
    print("  per any: " + ", ".join(f"{year} {count}" for year, count in years.items()))


def main():
    frame = prices.load()
    values = frame.to_numpy().ravel()
    by_hours = hampel_score(values, HOUR_HALF_WIDTH).reshape(frame.shape)
    by_days = np.column_stack([hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
                               for column in frame.columns])

    hour_median, hour_mad = window_median_mad(values, HOUR_HALF_WIDTH)
    day_parts = [window_median_mad(frame[column].to_numpy(), DAY_HALF_WIDTH)
                 for column in frame.columns]
    describe(frame.index, values, f"hores adjacents (±{HOUR_HALF_WIDTH} h)", by_hours,
             hour_median, hour_mad)
    describe(frame.index, values, f"mateixa hora (±{DAY_HALF_WIDTH} dies)", by_days,
             np.column_stack([part[0] for part in day_parts]).ravel(),
             np.column_stack([part[1] for part in day_parts]).ravel())

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

    instants = frame.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
        np.tile(hours, len(frame)), unit="h")
    bottom.scatter(instants, by_days.ravel(), s=1, color=SERIES, linewidths=0)
    bottom_marks = highest(by_days)
    bottom.scatter(instants[bottom_marks], by_days.ravel()[bottom_marks], s=12,
                   color=HIGHLIGHT_COLOR, zorder=3)
    bottom.set_xlim(instants[0], instants[-1])
    bottom.set_ylim(0, np.nanmax(by_days) * 1.5)
    style(bottom, f"Respecte de la mateixa hora dels dies del voltant (±{DAY_HALF_WIDTH} dies), "
          f"en vermell les {HIGHEST} més altes",
          "distància a la mediana, en MAD")

    figure.suptitle(f"Distància de cada preu horari als seus veïns, {frame.index[0].year} a "
                    f"{frame.index[-1].year}", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=DPI)
    print(f"desat a {OUTPUT}")
