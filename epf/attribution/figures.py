"""Les figures de SHAP d'una execució, dibuixades del que la passada d'atribució ha desat.

Cap figura torna a ajustar ni a atribuir res: llegeix el `.npz` que hi ha al costat de
les prediccions. Viuen aquí i no amb les figures que comparen pronòstics perquè són de
l'altra meitat d'una execució: aquelles diuen com de bo és un model i aquestes què
mira.

L'etapa serveix per experimentar i no encara per interpretar, així que se'n dibuixen
moltes i ja es decidirà quines es queden.
"""
import numpy as np
import pandas as pd

from epf.attribution import stored
from epf.attribution.stored import PRICE_UNITS
from epf.calendar import HOUR_COLUMNS
from epf.figure import (FOLLOWED, INK, MAGNITUDE, MUTED_INK, SURFACE, colorbar,
                        colours, plt, save, style)

# Per sota d'aquests mesos la figura del temps agrupa per setmana: amb un sol mes la
# línia té un punt i no es dibuixa res.
MONTHS_FOR_MONTHLY = 3

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


def per_hour_scale(importance, units):
    """Divisor de cada hora de lliurament, per posar-les totes en una escala llegible.

    Un model lineal escala l'objectiu de cada hora amb la seva pròpia mediana i
    escala, així que les contribucions de dues hores no són en les mateixes unitats i
    una sola escala de color les compara malament: mesurat sobre LEAR, les MAD d'un
    mateix dia es diferencien un factor 1,26 a la finestra de 1085 dies i 2,21 a la de
    49. Cada hora es divideix
    per la seva pròpia contribució més gran, de manera que el color és una part de
    l'hora i no una magnitud. Un arbre atribueix en EUR/MWh a totes les hores i sí
    que comparteix escala, així que allà el divisor és 1.
    """
    if units == PRICE_UNITS:
        return np.ones(importance.shape[1])
    largest = importance.max(axis=0)
    return np.where(largest > 0, largest, 1.0)


def _bar_label(units):
    return units if units == PRICE_UNITS else "part de l'hora"


def importance_by_lag(folder, label, stem, columns, importance, units):
    """Un plafó per retard: hora d'origen contra hora de lliurament."""
    lags, others = _blocks(columns)
    panels = len(lags) + bool(others)
    figure, axes = plt.subplots(1, panels, figsize=(5 * len(lags) + 3, 5), facecolor=SURFACE,
                                squeeze=False,
                                width_ratios=[1] * len(lags) + [0.35] * bool(others))
    # Els blocs de retard comparteixen escala, que és el que els fa comparables, i
    # la resta de columnes té la seva: en un arbre la columna de l'hora val el doble
    # que el retard més gran i, amb una escala sola, aplanava els quatre blocs.
    top = importance[[position for block in lags.values() for position in block]].max()
    for axis, (lag, positions) in zip(axes[0], sorted(lags.items())):
        image = axis.imshow(importance[positions], cmap=MAGNITUDE, vmin=0, vmax=top,
                            origin="lower")
        axis.set_xlabel("hora del dia predit", color=MUTED_INK)
        axis.set_ylabel(f"hora de fa {lag} dies", color=MUTED_INK)
        style(axis, f"preu de fa {lag} dies")
    # La barra dels retards es penja de l'últim dels seus plafons: penjada de tots
    # alhora, `tight_layout` no la sap col·locar i la dibuixa a sobre d'un.
    colorbar(figure, image, axes[0][len(lags) - 1], _bar_label(units))
    if others:
        axis = axes[0][-1]
        rest = axis.imshow(importance[others], cmap=MAGNITUDE, vmin=0,
                           aspect="auto", origin="lower")
        axis.set_yticks(range(len(others)), [columns[position] for position in others],
                        fontsize=6)
        axis.set_xlabel("hora del dia predit", color=MUTED_INK)
        style(axis, "la resta, amb la seva escala")
        colorbar(figure, rest, axis, _bar_label(units))
    figure.suptitle(f"{label}: importància de cada retard", color=INK, x=0.01, ha="left")
    save(figure, folder, f"{stem}_lags")


def importance_by_column(folder, label, stem, columns, importance, units):
    """Totes les columnes, en l'ordre en què les construeix la matriu."""
    figure, axis = plt.subplots(figsize=(10, 18), facecolor=SURFACE)
    image = axis.imshow(importance, cmap=MAGNITUDE, aspect="auto", origin="upper")
    axis.set_xticks(range(len(HOUR_COLUMNS)), HOUR_COLUMNS, rotation=90, fontsize=6)
    axis.set_yticks(range(len(columns)), columns, fontsize=4)
    axis.set_xlabel("hora del dia predit", color=MUTED_INK)
    colorbar(figure, image, axis, _bar_label(units))
    style(axis, f"{label}: totes les columnes")
    save(figure, folder, f"{stem}_columns")


def importance_totals(folder, label, stem, columns, importance, units):
    """Un retard per fila, sumant les hores d'origen.

    Sense el divisor per hora de les altres figures: cada fila és la suma de 24
    columnes i dividir-la per la contribució més gran d'una sola columna donaria
    una part de l'hora més gran que u. Es queda en les unitats desades, de manera
    que en un model lineal les hores d'aquesta figura no es comparen entre elles.
    """
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
    colorbar(figure, image, axis, units)
    style(axis, f"{label}: importància per retard")
    save(figure, folder, f"{stem}_totals")


def importance_over_time(folder, label, stem, columns, values, days, scale, units):
    """Les columnes més importants al llarg del temps, mes a mes o, si el període
    és curt, setmana a setmana.

    La mitjana és sobre les hores del dia, que en un model lineal no són en les
    mateixes unitats, de manera que cada hora es divideix pel seu divisor abans de
    mitjanar.
    """
    averaged = (np.abs(values) / scale).mean(axis=2)
    order = averaged.mean(axis=0).argsort()[::-1][:FOLLOWED]
    when = pd.DatetimeIndex(days)
    by_month = when.to_period("M").nunique() >= MONTHS_FOR_MONTHLY
    groups = when.to_period("M" if by_month else "W")
    figure, axis = plt.subplots(figsize=(14, 4.5), facecolor=SURFACE)
    for colour, position in colours(order):
        series = pd.Series(averaged[:, position], index=groups).groupby(level=0).mean()
        axis.plot(series.index.to_timestamp(), series.to_numpy(), color=colour,
                  label=columns[position])
    axis.set_ylabel(f"mitjana de |SHAP| ({_bar_label(units)})", color=MUTED_INK)
    axis.legend(frameon=False, labelcolor=MUTED_INK, ncols=4, fontsize=8)
    step = "mes a mes" if by_month else "setmana a setmana"
    style(axis, f"{label}: les {FOLLOWED} columnes més importants {step}")
    save(figure, folder, f"{stem}_over_time")


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
    style(axis, f"{label}: concentració, una línia per hora de lliurament")
    save(figure, folder, f"{stem}_concentration")


def draw(folder, name):
    """Les cinc figures de cada membre de l'atribució d'un pronòstic."""
    days, attribution = stored.load(folder, name)
    columns, units = attribution.columns, attribution.units
    for number, member in enumerate(attribution.members):
        # Una família d'un sol model no té cap finestra de què parlar al títol.
        several = len(attribution.members) > 1
        label = f"{name}, finestra {member}" if several else name
        stem = f"shap_{name}_{member}" if several else f"shap_{name}"
        importance = np.abs(attribution.values[:, number]).mean(axis=0)
        scale = per_hour_scale(importance, units)
        importance_by_lag(folder, label, stem, columns, importance / scale, units)
        importance_by_column(folder, label, stem, columns, importance / scale, units)
        importance_totals(folder, label, stem, columns, importance, units)
        importance_over_time(folder, label, stem, columns,
                             attribution.values[:, number], days, scale, units)
        importance_concentration(folder, label, stem, importance)
