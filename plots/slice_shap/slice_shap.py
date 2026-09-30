"""Importància mitjana de cada variable segons SHAP, en tres disposicions.

Cada dia del període es prediu amb un bosc reajustat només amb dies anteriors, i
s'explica amb el seu propi bosc: l'atribució d'un dia surt del model que el va
predir, no d'un model ajustat una vegada. Es guarda la mitjana de |SHAP| sobre tots
els dies.

L'esperat és una diagonal: l'hora h del dia que es prediu recolzada sobre la mateixa
hora del dia anterior. Si no hi és, el més probable no és cap fet del mercat sinó un
desalineament entre la matriu i el model.
"""
import argparse
import datetime
import pathlib
import time

import matplotlib
import numpy as np
import shap

from epf.config.market import HOURS_PER_DAY
from epf.config.split import TEST_PERIODS
from epf.data import prices
from epf.data.calendar import HOUR_COLUMNS
from epf.features import matrix
from epf.features.matrix import DAY_OF_WEEK, PRICE_LAGS
from epf.models import forest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = pathlib.Path(__file__).parent

INK = "#0b0b0b"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Escala seqüencial d'un sol to, de clar a fosc, a partir del blau del projecte:
# la magnitud es llegeix igual en color i en escala de grisos impresa.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [SURFACE, "#c3d9f2", "#6fa4e0", "#2a78d6", "#143f73"])

HOURLY_TEST = TEST_PERIODS["hourly"]
DPI = 300


def day(text):
    return datetime.date.fromisoformat(text)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--from", dest="first", type=day, default=HOURLY_TEST[0])
parser.add_argument("--to", dest="last", type=day, default=HOURLY_TEST[1])
parser.add_argument("--every", type=int, default=1, help="pren un dia de cada N")
parser.add_argument("--window", type=int, default=364 * 3, help="dies de calibratge")
parser.add_argument("--scale", choices=("shared", "panel"), default="shared",
                    help="una escala de color per a tots els retards o una per plafó")
args = parser.parse_args()

X, y = matrix.build(prices.load())
days = X.index[(X.index >= str(args.first)) & (X.index <= str(args.last))][::args.every]
print(f"{len(days)} dies, de {days[0].date()} a {days[-1].date()}, "
      f"finestra de {args.window} dies")

started = time.perf_counter()
total = np.zeros((X.shape[1], HOURS_PER_DAY))
for number, predicted_day in enumerate(days, start=1):
    fitted = forest.fit(X, y, predicted_day, window=args.window)
    values = shap.TreeExplainer(fitted).shap_values(X.loc[[predicted_day]])
    total += np.abs(values[0])
    elapsed = time.perf_counter() - started
    print(f"  {predicted_day.date()}  {number}/{len(days)}, {elapsed / number:.1f} s/dia",
          flush=True)

importance = total / len(days)
lags = importance[:-1].reshape(len(PRICE_LAGS), HOURS_PER_DAY, HOURS_PER_DAY)
week = importance[-1]

subtitle = (f"mitjana de |SHAP| sobre {len(days)} dies, "
            f"{days[0].date()} a {days[-1].date()}")


def style(axis, title):
    axis.set_title(title, color=INK, loc="left", fontsize=10)
    axis.tick_params(colors=MUTED_INK, length=0)
    for side in axis.spines.values():
        side.set_color(GRID)


def colorbar(figure, image, axis):
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    bar.set_label("EUR/MWh", color=MUTED_INK)
    bar.ax.tick_params(colors=MUTED_INK, length=0)
    bar.outline.set_color(GRID)


def save(figure, name):
    output = HERE / f"{name}.png"
    figure.tight_layout()
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
    print(f"desat a {output}")


# Una: un plafó per retard, hora d'origen contra hora de lliurament.
figure, axes = plt.subplots(1, len(PRICE_LAGS) + 1, figsize=(23, 5), facecolor=SURFACE,
                            width_ratios=[1] * len(PRICE_LAGS) + [0.3])
shared = args.scale == "shared"
top = lags.max()
for axis, lag, block in zip(axes, PRICE_LAGS, lags):
    limit = top if shared else block.max()
    image = axis.imshow(block, cmap=MAGNITUDE, vmin=0, vmax=limit, origin="lower")
    axis.set_xlabel("hora del dia predit", color=MUTED_INK)
    axis.set_ylabel(f"hora de fa {lag} dies", color=MUTED_INK)
    style(axis, f"preu de fa {lag} dies")
    if not shared:
        colorbar(figure, image, axis)
week_image = axes[-1].imshow(week.reshape(-1, 1), cmap=MAGNITUDE, vmin=0,
                             vmax=top if shared else week.max(), origin="lower")
axes[-1].set_xticks([])
axes[-1].set_ylabel("hora del dia predit", color=MUTED_INK)
style(axes[-1], "dia de\nla setmana")
colorbar(figure, week_image if not shared else image, axes[-1])
scale_said = "escala compartida" if shared else "una escala per plafó"
figure.suptitle(f"Importància de cada retard hora per hora, {scale_said}. {subtitle}",
                color=INK, x=0.01, ha="left")
save(figure, "per_lag" if shared else "per_lag_escalat")

# Dues: totes les 97 columnes, en l'ordre en què les construeix la matriu.
figure, axis = plt.subplots(figsize=(10, 18), facecolor=SURFACE)
image = axis.imshow(importance, cmap=MAGNITUDE, aspect="auto", origin="upper")
axis.set_xticks(range(HOURS_PER_DAY), HOUR_COLUMNS, rotation=90, fontsize=6)
names = [f"lag{lag}_{hour}" for lag in PRICE_LAGS for hour in HOUR_COLUMNS] + [DAY_OF_WEEK]
axis.set_yticks(range(len(names)), names, fontsize=4)
axis.set_xlabel("hora del dia predit", color=MUTED_INK)
colorbar(figure, image, axis)
style(axis, f"Totes les columnes. {subtitle}")
save(figure, "per_column")

# Tres: un retard per fila, sumant les hores d'origen.
figure, axis = plt.subplots(figsize=(14, 3.2), facecolor=SURFACE)
collapsed = np.vstack([lags.sum(axis=1), week])
image = axis.imshow(collapsed, cmap=MAGNITUDE, aspect="auto", origin="upper")
axis.set_xticks(range(HOURS_PER_DAY), HOUR_COLUMNS, rotation=90, fontsize=7)
axis.set_yticks(range(len(PRICE_LAGS) + 1),
                [f"preu de fa {lag} dies" for lag in PRICE_LAGS] + ["dia de la setmana"])
axis.set_xlabel("hora del dia predit", color=MUTED_INK)
colorbar(figure, image, axis)
style(axis, f"Importància de cada retard, sumant les hores d'origen. {subtitle}")
save(figure, "per_lag_total")
