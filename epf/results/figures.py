"""Les figures que comparen pronòstics, dibuixades del que l'execució va desar.

Cap figura torna a ajustar res: llegeix els CSV del disc. Les de SHAP són a
`attribution/figures.py`: aquestes diuen com de bo és un pronòstic i aquelles què
mira el model que el va fer.

L'etapa serveix per experimentar i no encara per interpretar, així que se'n dibuixen
moltes i ja es decidirà quines es queden.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from epf.calendar import HOUR_COLUMNS
from epf.evaluation import (curve_correlations, error_dispersion,
                            extrema_price_deviations, mae_per_hour, rmae_per_hour)
from epf.figure import (INK, MAGNITUDE, MUTED_INK, SURFACE, colorbar, colours,
                        plt, save, style)
from epf.models.naive import naive
from epf.results import run_files

# Escala de dos tons per a una correlació, que va de -1 a 1 al voltant del zero. És
# d'aquesta figura i de cap altra.
SIGNED = LinearSegmentedColormap.from_list(
    "signed", ["#8c3c12", "#d6622a", SURFACE, "#2a78d6", "#143f73"])


def _chained(frame):
    """Una taula de dia per fila i hora per columna, en una sola sèrie horària.

    Sobre una graella horària completa del primer dia a l'últim, de manera que un
    dia que no hi és queda absent i la línia s'hi trenca en lloc de travessar el
    buit: una passada mostrejada no s'ha de poder llegir com una sèrie contínua.
    """
    hours = pd.DatetimeIndex([day + pd.Timedelta(hours=hour)
                              for day in frame.index for hour in range(len(HOUR_COLUMNS))])
    series = pd.Series(frame.to_numpy().reshape(-1), index=hours)
    return series.reindex(pd.date_range(hours[0], hours[-1], freq="h"))


def prices_against_forecasts(folder, forecasts, actual):
    """El preu real i el que cada pronòstic va dir, hora rere hora."""
    figure, axis = plt.subplots(figsize=(16, 4.5), facecolor=SURFACE)
    axis.plot(_chained(actual), color=INK, linewidth=0.7, label="preu real")
    for colour, (name, frame) in colours(sorted(forecasts.items())):
        axis.plot(_chained(frame), color=colour, linewidth=0.6, alpha=0.8, label=name)
    axis.set_ylabel("EUR/MWh", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    style(axis, "Preu real i predit")
    save(figure, folder, "price")


def error_by_hour(folder, forecasts, actual, weekly):
    """Error de cada hora de lliurament, en brut i relatiu a la predicció ingènua.

    L'error absolut d'una hora es llegeix malament tot sol, perquè barreja com de
    difícil és l'hora amb com de cara és. El segon plafó el divideix pel que hi erra
    la predicció ingènua setmanal, que és com de difícil és.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=SURFACE)
    panels = ((lambda frame: mae_per_hour(frame, actual),
               "MAE (EUR/MWh)", "Error per hora de lliurament"),
              (lambda frame: rmae_per_hour(frame, actual, weekly),
               "MAE sobre el de la predicció ingènua", "Sense l'efecte de la dificultat"))
    for axis, (measure, label, title) in zip(axes, panels):
        for colour, (name, frame) in colours(sorted(forecasts.items())):
            axis.plot(range(len(HOUR_COLUMNS)), measure(frame).to_numpy(), color=colour,
                      marker="o", markersize=3, label=name)
        axis.set_xticks(range(len(HOUR_COLUMNS)), HOUR_COLUMNS, rotation=90, fontsize=7)
        axis.set_ylabel(label, color=MUTED_INK)
        style(axis, title)
    axes[-1].axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    save(figure, folder, "error_by_hour")


def curve_correlation_over_time(folder, forecasts, actual):
    """Corr-f dia a dia: com de bé reprodueix cada pronòstic la forma del dia.

    La correlació de Spearman entre les 24 hores predites i les reals de cada dia, de
    Maciejowska, Lipiecki i Uniejewski (2026). Un model pot errar el nivell del dia
    sencer i encertar-ne la forma, i això no es veu enlloc en un MAE.
    """
    figure, axes = plt.subplots(1, 2, figsize=(18, 4), facecolor=SURFACE,
                                width_ratios=[2.5, 1])
    for colour, (name, frame) in colours(sorted(forecasts.items())):
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
    style(axes[0], "Forma del dia encertada, dia a dia, amb la mitjana entre parèntesis")
    style(axes[1], "repartiment")
    save(figure, folder, "curve_correlation")


def daily_spread(folder, forecasts, actual):
    """La forquilla de cada dia i la que cada pronòstic hauria capturat.

    MPD de Maciejowska, Lipiecki i Uniejewski (2026) és la distància entre les dues
    línies: el que es perd per col·locar el mínim i el màxim del dia a les hores
    equivocades. Un model pot tenir un MAE excel·lent i deixar-se la forquilla sencera.
    """
    days = next(iter(forecasts.values())).index.intersection(actual.index)
    spread = actual.loc[days].max(axis="columns") - actual.loc[days].min(axis="columns")

    figure, axes = plt.subplots(1, 2, figsize=(18, 4), facecolor=SURFACE,
                                width_ratios=[2.5, 1])
    axes[0].plot(spread.index, spread.to_numpy(), color=INK, linewidth=0.8,
                 label="forquilla real")
    for colour, (name, frame) in colours(sorted(forecasts.items())):
        lost = extrema_price_deviations(frame, actual)
        axes[0].plot(lost.index, (spread - lost).to_numpy(), color=colour, linewidth=0.7,
                     alpha=0.8, label=f"{name} (MPD {lost.mean():.2f})")
        axes[1].hist(lost.to_numpy(), bins=40, histtype="step", color=colour)
    axes[0].set_ylabel("EUR/MWh", color=MUTED_INK)
    axes[0].legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    axes[1].set_xlabel("MPD del dia (EUR/MWh)", color=MUTED_INK)
    style(axes[0], "Forquilla del dia i la que el pronòstic hauria capturat")
    style(axes[1], "repartiment del que es perd")
    save(figure, folder, "daily_spread")


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
    colorbar(figure, first, axes[0], "(EUR/MWh)²")
    style(axes[0], "Segons moments dels errors")
    second = axes[1].imshow(correlation, cmap=SIGNED, vmin=-1, vmax=1, origin="lower")
    colorbar(figure, second, axes[1], "correlació")
    style(axes[1], "La mateixa matriu, en correlació")
    for axis in axes:
        axis.set_xlabel("hora de lliurament", color=MUTED_INK)
        axis.set_ylabel("hora de lliurament", color=MUTED_INK)
    figure.suptitle(f"{name}: errors dins del dia. Cov-e = "
                    f"{error_dispersion(frame, actual):.2f} sobre {len(days)} dies, "
                    "com més baix millor", color=INK, x=0.01, ha="left")
    save(figure, folder, f"error_covariance_{name}")


def error_over_time(folder, forecasts, actual):
    """Error absolut mitjà al llarg del temps, per setmanes o, si els dies no són
    consecutius, per mesos.

    Una passada mostrejada cada vuit dies té setmanes amb un sol dia i setmanes
    sense cap, de manera que una mitjana setmanal no és una mitjana: el pas surt
    dels dies que hi ha i no d'un calendari triat per endavant.
    """
    days = next(iter(forecasts.values())).index
    consecutive = len(days) < 2 or days.to_series().diff().dt.days.median() == 1
    step, period = ("W", "setmana a setmana") if consecutive else ("ME", "mes a mes")
    figure, axis = plt.subplots(figsize=(14, 4), facecolor=SURFACE)
    for colour, (name, frame) in colours(sorted(forecasts.items())):
        daily = (frame - actual.loc[frame.index]).abs().mean(axis="columns")
        grouped = daily.resample(step).mean()
        axis.plot(grouped.index, grouped.to_numpy(), color=colour, label=name)
    axis.set_ylabel("MAE (EUR/MWh)", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    style(axis, f"Error {period}")
    save(figure, folder, "error_over_time")


def error_distribution(folder, forecasts, actual):
    """Repartiment de l'error amb signe: on es queda curt i on es passa."""
    figure, axis = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    for colour, (name, frame) in colours(sorted(forecasts.items())):
        signed = (frame - actual.loc[frame.index]).to_numpy().reshape(-1)
        axis.hist(signed, bins=120, histtype="step", color=colour, label=name)
    axis.set_xlabel("predicció menys preu real (EUR/MWh)", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=3)
    style(axis, "Repartiment de l'error")
    save(figure, folder, "error_distribution")


def shared(folder, forecasts, actual):
    """Les figures que comparen pronòstics entre ells.

    Es dibuixen allà on viuen els que es comparen: dins d'una família són els seus,
    i al directori de la configuració són els de totes les famílies i la combinació
    alhora, que és el que de fet s'analitza. Qui crida tria què hi entra, perquè la
    combinació la munta l'etapa i aquest mòdul no en sap res.
    """
    weekly = naive.weekly_frame(actual)
    actual = actual.loc[next(iter(forecasts.values())).index]
    prices_against_forecasts(folder, forecasts, actual)
    error_by_hour(folder, forecasts, actual, weekly)
    error_over_time(folder, forecasts, actual)
    error_distribution(folder, forecasts, actual)
    curve_correlation_over_time(folder, forecasts, actual)
    daily_spread(folder, forecasts, actual)


def draw(folder, actual):
    """Les figures d'una família ja desada que es poden dibuixar de les prediccions.

    Les de SHAP no: demanen el `.npz`, que escriu la passada d'atribució, i les dibuixa
    ella.
    """
    forecasts = run_files.load_forecasts(folder)
    drawn = {name: forecasts[name] for name in run_files.headlines(folder, forecasts)}
    shared(folder, drawn, actual)
    actual = actual.loc[next(iter(forecasts.values())).index]
    for name, frame in sorted(drawn.items()):
        error_covariance(folder, name, frame, actual)
