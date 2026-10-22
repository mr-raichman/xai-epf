"""L'estudi d'outliers d'una sèrie: com s'allunya cada valor dels seus veïns.

L'informe no refusa res i no porta cap llindar.
"""

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from numpy.lib.stride_tricks import sliding_window_view

from epf import paths
from epf.contract import figure
from epf.contract.figure import plt
from epf.contract.matrix import HOUR
from epf.parameters import HOURS_PER_DAY

# Veïns d'un valor, a cada costat: les hores adjacents de la sèrie cronològica, i la
# mateixa hora dels dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# Les puntuacions més altes que es desen de cada neighbourhood; no és cap llindar.
HIGHEST = 100

# On l'eix de les puntuacions passa de lineal a logarítmic: la majoria són inferiors a
# 10 i les més altes arriben als milers.
LINEAR_UP_TO = 1

# La mida de les puntuacions marcades, la d'un punt d'una capa que no és densa.
MARK_SIZE = figure.MARKER_SIZE

# L'alçada de la figura, en mm, i fins on arriba l'eix de les puntuacions de cada
# panell, en vegades la seva puntuació més alta.
HEIGHT = 91
AXIS_TOP = 1.5

# El nom de la taula i de la figura de l'estudi d'una sèrie, sense l'extensió.
OUTLIERS = "outliers_{series}"


def window_median_mad(values, half_width):
    """Torna la mediana i la MAD de la finestra centrada en cada valor.

    La finestra té `half_width` valors a cada costat, i els extrems, sense finestra
    completa, valen NaN.
    """
    values = np.asarray(values, dtype=float)
    windows = sliding_window_view(values, 2 * half_width + 1)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    edge = np.full(half_width, np.nan)
    return np.concatenate([edge, median, edge]), np.concatenate([edge, mad, edge])


def hampel_score(values, half_width):
    """Torna la distància de cada valor a la mediana de la seva finestra, en MADs.

    Una finestra de MAD zero puntua 0, i els extrems sense finestra completa, NaN.
    """
    values = np.asarray(values, dtype=float)
    median, mad = window_median_mad(values, half_width)
    distance = np.abs(values - median)
    score = np.divide(distance, mad, out=np.zeros_like(distance), where=mad > 0)
    score[np.isnan(median)] = np.nan
    return score


def highest(scores):
    """Torna les posicions de les `HIGHEST` puntuacions més altes, sense les NaN.

    Són posicions de `scores` aplanat, de la puntuació més baixa a la més alta.
    """
    flat = np.nan_to_num(scores.ravel(), nan=-np.inf)
    return np.argsort(flat)[-HIGHEST:]


def report(name, frame):
    """Puntua cada valor d'una sèrie contra els seus veïns i en desa les més altes.

    De cada neighbourhood, les hores adjacents de la sèrie cronològica, `hours`, i la
    mateixa hora dels dies del voltant, `days`, desa les `HIGHEST` puntuacions més altes
    a `outliers_<sèrie>.csv`, a la carpeta de la sèrie, que ja hi ha de ser, cada una
    amb el valor i la mediana i la MAD de la seva finestra. Torna totes les puntuacions
    de cada neighbourhood, en dies per hores, per dibuixar-les.
    """
    values = frame.to_numpy().ravel()
    by_hours = hampel_score(values, HOUR_HALF_WIDTH).reshape(frame.shape)
    by_days = np.column_stack(
        [
            hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
            for column in frame.columns
        ]
    )
    hour_median, hour_mad = window_median_mad(values, HOUR_HALF_WIDTH)
    day_parts = [
        window_median_mad(frame[column].to_numpy(), DAY_HALF_WIDTH)
        for column in frame.columns
    ]
    views = (
        ("hours", by_hours, hour_median, hour_mad),
        (
            "days",
            by_days,
            np.column_stack([part[0] for part in day_parts]).ravel(),
            np.column_stack([part[1] for part in day_parts]).ravel(),
        ),
    )
    days = frame.index.repeat(HOURS_PER_DAY).strftime("%Y-%m-%d")
    hours = np.tile(np.arange(HOURS_PER_DAY), len(frame))
    rows = []
    for view, scores, median, mad in views:
        marks = highest(scores)
        marks = marks[np.argsort(-scores.ravel()[marks], kind="stable")]
        rows.append(
            pd.DataFrame(
                {
                    "neighbourhood": view,
                    "day": days[marks],
                    "hour": hours[marks],
                    "value": values[marks],
                    "median": median[marks],
                    "mad": mad[marks],
                    "score": scores.ravel()[marks],
                }
            )
        )
    pd.concat(rows).to_csv(
        paths.DATA_FIGURES / name / f"{OUTLIERS.format(series=name)}.csv",
        index=False,
        lineterminator="\n",
        encoding="utf-8",
    )
    return by_hours, by_days


def _style(axis, title):
    """Dona a uns eixos el títol i l'escala vertical."""
    axis.set_title(title)
    axis.set_yscale("symlog", linthresh=LINEAR_UP_TO)


def draw(name, title, frame, by_hours, by_days):
    """Dibuixa les puntuacions d'outliers d'una sèrie a `outliers_<sèrie>.svg`.

    `title` és el nom de la sèrie a les figures. A la carpeta de la sèrie: les de les
    hores adjacents, `by_hours`, un dia per línia, i les de la mateixa hora dels dies
    del voltant, `by_days`, en el temps, amb les `HIGHEST` més altes de cada
    neighbourhood en vermell.
    """
    panels, (top, bottom) = plt.subplots(2, 1, figsize=figure.size(HEIGHT))

    hours = np.arange(HOURS_PER_DAY)
    top.add_collection(
        LineCollection(
            [np.column_stack([hours, day]) for day in by_hours],
            colors=figure.OTHER_COLOUR,
            **figure.line_style(by_hours.size, overlapping=True),
        )
    )
    top_marks = highest(by_hours)
    top.scatter(
        top_marks % HOURS_PER_DAY,
        by_hours.ravel()[top_marks],
        s=MARK_SIZE**2,
        color=figure.MARK_COLOUR,
        zorder=3,
    )
    top.set_xlim(0, HOURS_PER_DAY - 1)
    top.set_ylim(0, np.nanmax(by_hours) * AXIS_TOP)
    top.set_xticks(hours)
    top.set_xlabel(figure.CALENDAR_LABELS[HOUR])
    _style(
        top,
        f"Respecte de les hores adjacents (±{HOUR_HALF_WIDTH} h), un dia per línia, "
        f"en vermell les {HIGHEST} més altes",
    )

    instants = frame.index.repeat(HOURS_PER_DAY) + pd.to_timedelta(
        np.tile(hours, len(frame)), unit="h"
    )
    bottom.scatter(
        instants,
        by_days.ravel(),
        color=figure.OTHER_COLOUR,
        **figure.point_style(by_days.size),
    )
    bottom_marks = highest(by_days)
    bottom.scatter(
        instants[bottom_marks],
        by_days.ravel()[bottom_marks],
        s=MARK_SIZE**2,
        color=figure.MARK_COLOUR,
        zorder=3,
    )
    bottom.set_xlim(instants[0], instants[-1])
    bottom.set_ylim(0, np.nanmax(by_days) * AXIS_TOP)
    _style(
        bottom,
        f"Respecte de la mateixa hora dels dies del voltant (±{DAY_HALF_WIDTH} "
        f"dies), en vermell les {HIGHEST} més altes",
    )

    panels.suptitle(
        f"{title}\ndistància de cada valor horari als seus veïns, de "
        f"{frame.index[0].year} a {frame.index[-1].year}"
    )
    panels.supylabel(
        figure.unit_label("distància a la mediana", "MAD"),
        fontsize=figure.AXIS_TEXT_SIZE,
    )
    figure.save(panels, paths.DATA_FIGURES / name, OUTLIERS.format(series=name))
