"""Figures d'una execució de l'etapa E1, dibuixades del que l'execució va desar.

Cap figura torna a ajustar res: l'atribució de cada dia surt del model que el va
predir i ja és al directori. L'etapa serveix per experimentar i no encara per
interpretar, així que se'n dibuixen moltes i ja es decidirà quines es queden.
"""
import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd

from epf import run as run_files
from epf.data.calendar import HOUR_COLUMNS
from epf.e1 import features
from epf.evaluation import (curve_correlations, error_dispersion, mae_per_hour,
                            mae_per_hour_over_price, rmae_per_hour)
from epf.models import naive as naive_family
from epf.models import attribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Escala seqüencial d'un sol to: la magnitud es llegeix igual en color i en escala
# de grisos impresa.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [SURFACE, "#c3d9f2", "#6fa4e0", "#2a78d6", "#143f73"])

# Escala de dos tons per a una correlació, que va de -1 a 1 al voltant del zero.
SIGNED = LinearSegmentedColormap.from_list(
    "signed", ["#8c3c12", "#d6622a", SURFACE, "#2a78d6", "#143f73"])
LINES = ("#2a78d6", "#d6622a", "#2ba86a", "#9b4fd6", "#c0392b", "#117a8b")

DPI = 300

# Columnes que es segueixen al llarg del temps i de les quals es mira la
# concentració: més no es llegeixen en una llegenda.
FOLLOWED = 8


def _style(axis, title=None):
    if title:
        axis.set_title(title, color=INK, loc="left", fontsize=10)
    axis.tick_params(colors=MUTED_INK, length=0)
    for side in axis.spines.values():
        side.set_color(GRID)


def _colorbar(figure, image, axis, label):
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    bar.set_label(label, color=MUTED_INK)
    bar.ax.tick_params(colors=MUTED_INK, length=0)
    bar.outline.set_color(GRID)


def _save(figure, folder, name):
    figure.tight_layout()
    figure.savefig(folder / f"{name}.png", dpi=DPI)
    plt.close(figure)


def _chained(frame):
    """Una taula de dia per fila i hora per columna, en una sola sèrie horària."""
    hours = pd.DatetimeIndex([day + pd.Timedelta(hours=hour)
                              for day in frame.index for hour in range(len(HOUR_COLUMNS))])
    return pd.Series(frame.to_numpy().reshape(-1), index=hours)


def _blocks(columns):
    """Separa les columnes en blocs de retard i la resta, per posició."""
    lags, others = {}, []
    for position, name in enumerate(columns):
        head, _, tail = name.partition("_")
        if head.startswith("lag") and tail in HOUR_COLUMNS:
            lags.setdefault(int(head[3:]), []).append(position)
        else:
            others.append(position)
    return lags, others


def prices_against_forecasts(folder, forecasts, actual):
    """El preu real i el que cada pronòstic va dir, hora rere hora."""
    figure, axis = plt.subplots(figsize=(16, 4.5), facecolor=SURFACE)
    axis.plot(_chained(actual), color=INK, linewidth=0.7, label="preu real")
    for colour, (name, frame) in zip(LINES, sorted(forecasts.items())):
        axis.plot(_chained(frame), color=colour, linewidth=0.6, alpha=0.8, label=name)
    axis.set_ylabel("EUR/MWh", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    _style(axis, "Preu real i predit")
    _save(figure, folder, "price")


def error_by_hour(folder, forecasts, actual, naive):
    """Error de cada hora de lliurament, en brut i de dues maneres sense el nivell.

    El preu mitjà de les hores puntes és el doble que el de les valls i l'error absolut
    el segueix. El segon plafó el divideix pel preu de l'hora, que treu el nivell i
    també part de la variabilitat que sí que ha produït l'error; el tercer el divideix
    pel que hi erra la predicció ingènua setmanal, que és com de difícil és l'hora.
    """
    figure, axes = plt.subplots(1, 3, figsize=(21, 4), facecolor=SURFACE)
    panels = ((lambda frame: mae_per_hour(frame, actual),
               "MAE (EUR/MWh)", "Error per hora de lliurament"),
              (lambda frame: mae_per_hour_over_price(frame, actual),
               "MAE sobre el preu mitjà de l'hora", "Sense l'efecte del nivell"),
              (lambda frame: rmae_per_hour(frame, actual, naive),
               "MAE sobre el de la predicció ingènua", "Sense l'efecte de la dificultat"))
    for axis, (measure, label, title) in zip(axes, panels):
        for colour, (name, frame) in zip(LINES, sorted(forecasts.items())):
            axis.plot(range(len(HOUR_COLUMNS)), measure(frame).to_numpy(), color=colour,
                      marker="o", markersize=3, label=name)
        axis.set_xticks(range(len(HOUR_COLUMNS)), HOUR_COLUMNS, rotation=90, fontsize=7)
        axis.set_ylabel(label, color=MUTED_INK)
        _style(axis, title)
    axes[-1].axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, labelcolor=MUTED_INK)
    _save(figure, folder, "error_by_hour")


def curve_correlation_over_time(folder, forecasts, actual):
    """Corr-f dia a dia: com de bé reprodueix cada pronòstic la forma del dia.

    La correlació de Spearman entre les 24 hores predites i les reals de cada dia, de
    Maciejowska, Lipiecki i Uniejewski (2026). Un model pot errar el nivell del dia
    sencer i encertar-ne la forma, i això no es veu enlloc en un MAE.
    """
    figure, axes = plt.subplots(1, 2, figsize=(18, 4), facecolor=SURFACE,
                                width_ratios=[2.5, 1])
    for colour, (name, frame) in zip(LINES, sorted(forecasts.items())):
        daily = curve_correlations(frame, actual)
        axes[0].plot(daily.index, daily.to_numpy(), color=colour, linewidth=0.7, alpha=0.8,
                     label=f"{name} ({daily.mean():.3f})")
        axes[1].hist(daily.dropna().to_numpy(), bins=40, histtype="step", color=colour,
                     orientation="horizontal")
    for axis in axes:
        axis.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("correlació de Spearman del dia", color=MUTED_INK)
    axes[0].legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    axes[1].set_xlabel("dies", color=MUTED_INK)
    _style(axes[0], "Forma del dia encertada, dia a dia, amb la mitjana entre parèntesis")
    _style(axes[1], "repartiment")
    _save(figure, folder, "curve_correlation")


def error_covariance(folder, name, frame, actual):
    """Com es reparteixen els errors d'un dia entre les 24 hores.

    A l'esquerra la matriu de segons moments dels errors, de la qual Cov-e és el
    logaritme del determinant; a la dreta la mateixa matriu en forma de correlació, que
    és on es llegeix si el model s'equivoca el dia sencer en bloc o hora per hora.
    Equivocar-se en bloc conserva la forma del dia i baixa Cov-e.
    """
    days = frame.index.intersection(actual.index)
    errors = (frame.loc[days] - actual.loc[days]).to_numpy()
    moment = errors.T @ errors / len(errors)
    deviation = np.sqrt(np.diag(moment))
    correlation = moment / np.outer(deviation, deviation)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)
    first = axes[0].imshow(moment, cmap=MAGNITUDE, origin="lower")
    _colorbar(figure, first, axes[0], "(EUR/MWh)²")
    _style(axes[0], "Segons moments dels errors")
    second = axes[1].imshow(correlation, cmap=SIGNED, vmin=-1, vmax=1, origin="lower")
    _colorbar(figure, second, axes[1], "correlació")
    _style(axes[1], "La mateixa matriu, en correlació")
    for axis in axes:
        axis.set_xlabel("hora de lliurament", color=MUTED_INK)
        axis.set_ylabel("hora de lliurament", color=MUTED_INK)
    figure.suptitle(f"{name}: errors dins del dia. Cov-e = "
                    f"{error_dispersion(frame, actual):.2f} sobre {len(days)} dies, "
                    "com més baix millor", color=INK, x=0.01, ha="left")
    _save(figure, folder, f"error_covariance_{name}")


def error_over_time(folder, forecasts, actual):
    """Error absolut mitjà setmana a setmana."""
    figure, axis = plt.subplots(figsize=(14, 4), facecolor=SURFACE)
    for colour, (name, frame) in zip(LINES, sorted(forecasts.items())):
        daily = (frame - actual.loc[frame.index]).abs().mean(axis="columns")
        weekly = daily.resample("W").mean()
        axis.plot(weekly.index, weekly.to_numpy(), color=colour, label=name)
    axis.set_ylabel("MAE (EUR/MWh)", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK)
    _style(axis, "Error setmana a setmana")
    _save(figure, folder, "error_over_time")


def error_distribution(folder, forecasts, actual):
    """Repartiment de l'error amb signe: on es queda curt i on es passa."""
    figure, axis = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    for colour, (name, frame) in zip(LINES, sorted(forecasts.items())):
        signed = (frame - actual.loc[frame.index]).to_numpy().reshape(-1)
        axis.hist(signed, bins=120, histtype="step", color=colour, label=name)
    axis.set_xlabel("predicció menys preu real (EUR/MWh)", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK)
    _style(axis, "Repartiment de l'error")
    _save(figure, folder, "error_distribution")


def importance_by_lag(folder, label, stem, columns, importance, units):
    """Un plafó per retard: hora d'origen contra hora de lliurament."""
    lags, others = _blocks(columns)
    panels = len(lags) + bool(others)
    figure, axes = plt.subplots(1, panels, figsize=(5 * len(lags) + 2, 5), facecolor=SURFACE,
                                squeeze=False,
                                width_ratios=[1] * len(lags) + [0.35] * bool(others))
    top = importance.max()
    for axis, (lag, positions) in zip(axes[0], sorted(lags.items())):
        image = axis.imshow(importance[positions], cmap=MAGNITUDE, vmin=0, vmax=top,
                            origin="lower")
        axis.set_xlabel("hora del dia predit", color=MUTED_INK)
        axis.set_ylabel(f"hora de fa {lag} dies", color=MUTED_INK)
        _style(axis, f"preu de fa {lag} dies")
    if others:
        axis = axes[0][-1]
        image = axis.imshow(importance[others], cmap=MAGNITUDE, vmin=0, vmax=top,
                            aspect="auto", origin="lower")
        axis.set_yticks(range(len(others)), [columns[position] for position in others],
                        fontsize=6)
        axis.set_xlabel("hora del dia predit", color=MUTED_INK)
        _style(axis, "la resta de columnes")
    _colorbar(figure, image, axes[0][-1], units)
    figure.suptitle(f"{label}: importància de cada retard", color=INK, x=0.01, ha="left")
    _save(figure, folder, f"{stem}_lags")


def importance_by_column(folder, label, stem, columns, importance, units):
    """Totes les columnes, en l'ordre en què les construeix la matriu."""
    figure, axis = plt.subplots(figsize=(10, 18), facecolor=SURFACE)
    image = axis.imshow(importance, cmap=MAGNITUDE, aspect="auto", origin="upper")
    axis.set_xticks(range(len(HOUR_COLUMNS)), HOUR_COLUMNS, rotation=90, fontsize=6)
    axis.set_yticks(range(len(columns)), columns, fontsize=4)
    axis.set_xlabel("hora del dia predit", color=MUTED_INK)
    _colorbar(figure, image, axis, units)
    _style(axis, f"{label}: totes les columnes")
    _save(figure, folder, f"{stem}_columns")


def importance_totals(folder, label, stem, columns, importance, units):
    """Un retard per fila, sumant les hores d'origen."""
    lags, others = _blocks(columns)
    rows = [importance[positions].sum(axis=0) for _, positions in sorted(lags.items())]
    labels = [f"preu de fa {lag} dies" for lag in sorted(lags)]
    for position in others:
        rows.append(importance[position])
        labels.append(columns[position])
    figure, axis = plt.subplots(figsize=(14, 0.4 * len(rows) + 2), facecolor=SURFACE)
    image = axis.imshow(np.vstack(rows), cmap=MAGNITUDE, aspect="auto", origin="upper")
    axis.set_xticks(range(len(HOUR_COLUMNS)), HOUR_COLUMNS, rotation=90, fontsize=7)
    axis.set_yticks(range(len(rows)), labels, fontsize=7)
    axis.set_xlabel("hora del dia predit", color=MUTED_INK)
    _colorbar(figure, image, axis, units)
    _style(axis, f"{label}: importància per retard")
    _save(figure, folder, f"{stem}_totals")


def importance_over_time(folder, label, stem, columns, values, days, units):
    """Les columnes més importants, mes a mes."""
    monthly = np.abs(values).mean(axis=2)
    order = monthly.mean(axis=0).argsort()[::-1][:FOLLOWED]
    months = pd.DatetimeIndex(days).to_period("M")
    figure, axis = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    for position, colour in zip(order, LINES * 2):
        series = pd.Series(monthly[:, position], index=months).groupby(level=0).mean()
        axis.plot(series.index.to_timestamp(), series.to_numpy(), color=colour,
                  label=columns[position])
    axis.set_ylabel(f"mitjana de |SHAP| ({units})", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=4, fontsize=8)
    _style(axis, f"{label}: les {FOLLOWED} columnes més importants mes a mes")
    _save(figure, folder, f"{stem}_over_time")


def importance_concentration(folder, label, stem, importance):
    """Quina part de l'atribució total es concentra en les primeres columnes."""
    figure, axis = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for hour in range(importance.shape[1]):
        ordered = np.sort(importance[:, hour])[::-1]
        total = ordered.sum()
        if total:
            axis.plot(range(1, len(ordered) + 1), np.cumsum(ordered) / total,
                      color=MUTED_INK, linewidth=0.6, alpha=0.6)
    axis.set_xscale("log")
    axis.set_xlabel("columnes, de la més important a la menys", color=MUTED_INK)
    axis.set_ylabel("part acumulada de |SHAP|", color=MUTED_INK)
    _style(axis, f"{label}: concentració, una línia per hora de lliurament")
    _save(figure, folder, f"{stem}_concentration")


def draw(folder, actual):
    """Totes les figures d'una execució ja desada."""
    forecasts = run_files.load_forecasts(folder)
    naive = naive_family.weekly_frame(actual)
    actual = actual.loc[next(iter(forecasts.values())).index]

    prices_against_forecasts(folder, forecasts, actual)
    error_by_hour(folder, forecasts, actual, naive)
    error_over_time(folder, forecasts, actual)
    error_distribution(folder, forecasts, actual)
    curve_correlation_over_time(folder, forecasts, actual)
    for name, frame in sorted(forecasts.items()):
        error_covariance(folder, name, frame, actual)

    for name in forecasts:
        if not attribution.path(folder, name).exists():
            continue
        days, members, columns, units, values = attribution.load(folder, name)
        for number, member in enumerate(members):
            # Una família d'un sol model no té cap finestra de què parlar al títol.
            several = len(members) > 1
            label = f"{name}, finestra {member}" if several else name
            stem = f"shap_{name}_{member}" if several else f"shap_{name}"
            importance = np.abs(values[:, number]).mean(axis=0)
            importance_by_lag(folder, label, stem, columns, importance, units)
            importance_by_column(folder, label, stem, columns, importance, units)
            importance_totals(folder, label, stem, columns, importance, units)
            importance_over_time(folder, label, stem, columns, values[:, number], days, units)
            importance_concentration(folder, label, stem, importance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=pathlib.Path, help="directori d'una execució ja feta")
    folder = parser.parse_args().folder

    (_, actual), _ = features.load()
    draw(folder, actual)
    print(f"figures desades a {folder}")


if __name__ == "__main__":
    main()
