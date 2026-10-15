"""Com s'allunya cada valor d'una sèrie dels seus veïns, i quins són els més allunyats.

La puntuació, l'informe que la llegeix i la figura que la dibuixa viuen junts: la taula
que s'imprimeix i la figura han de ser dels mateixos números. L'informe no refusa res i
no porta cap llindar, i no n'hi pot haver: la puntuació és distància partit per MAD, i al
preu els seus extrems són moviments ordinaris dins de finestres de preu gairebé pla, no
moviments grans.
"""
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from numpy.lib.stride_tricks import sliding_window_view

from epf import paths
from epf.contract import figure
from epf.contract.figure import GRID, INK, MUTED_INK, SERIES, SURFACE, plt
from epf.parameters import HOURS_PER_DAY

# Veïns d'un valor: les hores adjacents de la sèrie cronològica, i la mateixa hora en
# els dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# Quantes de les puntuacions més altes s'informen. És quantes files s'imprimeixen i
# no un llindar: no decideix res sobre cap valor.
HIGHEST = 100

# La majoria de puntuacions són inferiors a 10 i les més altes arriben als milers:
# eix lineal fins a 1 i logarítmic a partir d'aquí.
LINEAR_UP_TO = 1

# Vermell de les puntuacions marcades, que és d'aquesta figura i de cap altra.
HIGHLIGHT_COLOR = "#e34948"


def window_median_mad(values, half_width):
    """Mediana i MAD de la finestra centrada en cada valor, amb `half_width` valors
    a cada costat. Els extrems, sense finestra completa, valen NaN."""
    values = np.asarray(values, dtype=float)
    windows = sliding_window_view(values, 2 * half_width + 1)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    edge = np.full(half_width, np.nan)
    return np.concatenate([edge, median, edge]), np.concatenate([edge, mad, edge])


def hampel_score(values, half_width):
    """Distància a la mediana de la finestra centrada, en MADs de la finestra.

    Una finestra de MAD zero puntua 0, i els extrems sense finestra completa, NaN.
    """
    values = np.asarray(values, dtype=float)
    median, mad = window_median_mad(values, half_width)
    distance = np.abs(values - median)
    score = np.divide(distance, mad, out=np.zeros_like(distance), where=mad > 0)
    score[np.isnan(median)] = np.nan
    return score


def highest(scores):
    """Posicions de les `HIGHEST` puntuacions més altes, sense comptar les NaN."""
    flat = np.nan_to_num(scores.ravel(), nan=-np.inf)
    return np.argsort(flat)[-HIGHEST:]


def _describe(days, values, label, scores, median, mad):
    """Resum de les puntuacions més altes: com és la finestra i on cauen."""
    marks = highest(scores)
    years = pd.Series(days.repeat(HOURS_PER_DAY)[marks].year).value_counts().sort_index()
    print(f"{label}, les {HIGHEST} puntuacions més altes:")
    print(f"  MAD de la finestra, mediana {np.median(mad[marks]):.2f}, "
          f"màxim {mad[marks].max():.2f}")
    print(f"  mediana de la finestra, mediana {np.median(median[marks]):.2f}")
    print(f"  distància del valor a la mediana, mediana "
          f"{np.median(np.abs(values - median)[marks]):.2f}")
    print("  per any: " + ", ".join(f"{year} {count}" for year, count in years.items()))


def report(name, frame):
    """Imprimeix les puntuacions de veïnatge més altes i les torna, per si es dibuixen.

    Les dues vistes: contra les hores adjacents de la sèrie cronològica, i contra la
    mateixa hora dels dies del voltant.
    """
    values = frame.to_numpy().ravel()
    by_hours = hampel_score(values, HOUR_HALF_WIDTH).reshape(frame.shape)
    by_days = np.column_stack([hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
                               for column in frame.columns])

    hour_median, hour_mad = window_median_mad(values, HOUR_HALF_WIDTH)
    day_parts = [window_median_mad(frame[column].to_numpy(), DAY_HALF_WIDTH)
                 for column in frame.columns]
    print(f"veïnatge de {name}:")
    _describe(frame.index, values, f"hores adjacents (±{HOUR_HALF_WIDTH} h)", by_hours,
              hour_median, hour_mad)
    _describe(frame.index, values, f"mateixa hora (±{DAY_HALF_WIDTH} dies)", by_days,
              np.column_stack([part[0] for part in day_parts]).ravel(),
              np.column_stack([part[1] for part in day_parts]).ravel())
    return by_hours, by_days


def _style(axis, title, ylabel):
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


def draw(name, frame, by_hours, by_days):
    """La figura de les puntuacions de veïnatge d'una sèrie, les que `report` ha imprès, a
    `outliers_<sèrie>.png`: les de les hores adjacents, un dia per línia, i les de la
    mateixa hora dels dies del voltant, en el temps, amb les més altes en vermell."""
    panels, (top, bottom) = plt.subplots(2, 1, figsize=(20, 12), facecolor=SURFACE)

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
    _style(top, f"Respecte de les hores adjacents (±{HOUR_HALF_WIDTH} h), un dia per línia, "
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
    _style(bottom, f"Respecte de la mateixa hora dels dies del voltant (±{DAY_HALF_WIDTH} "
           f"dies), en vermell les {HIGHEST} més altes",
           "distància a la mediana, en MAD")

    panels.suptitle(f"Distància de cada valor horari de {name} als seus veïns, "
                    f"{frame.index[0].year} a {frame.index[-1].year}", color=INK, x=0.01,
                    ha="left")
    figure.save(panels, paths.DATA_FIGURES, f"outliers_{name}")
