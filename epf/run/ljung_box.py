"""El correlograma dels errors de cada família d'una passada, del test de Ljung-Box."""

import json

import numpy as np

from epf import paths
from epf.contract import figure
from epf.parameters import DAYS_PER_WEEK
from epf.run import families
from epf.run.tables import PART_I_SECONDARY

# On cauen el 95 % de les autocorrelacions d'un soroll blanc: dins de ±1.96/√T, amb T la
# llargada de la sèrie, aquí els dies amb error (Hyndman i Athanasopoulos, 2021).
WHITE_NOISE_Z = 1.96

# Columnes de la figura, una família per panell, i l'alçada de cada panell entre la seva
# amplada.
AUTOCORRELATION_PER_ROW = 3
AUTOCORRELATION_ASPECT = 3 / 4


def report(where, numbers, colours):
    """Dibuixa el correlograma dels errors de cada família que té el test, a `where`.

    Dels números de la passada, `numbers`, com els torna `tables.write`: de cada
    família, les autocorrelacions de la sèrie cosida de totes les seeds, el seu valor p
    i els seus dies. Va a `autocorrelation.svg`, del color de cada família de
    `colours`; marca la distància de la partició i la setmana, i on cauria el 95 % de
    les autocorrelacions si l'error fos soroll blanc.
    """
    found = numbers[PART_I_SECONDARY]
    stitched = found[found["seed"].isna()]
    tested = set(stitched.loc[stitched["measure"] == "ljung_box_q", "family"])
    checked = [family for family in families.EVERY if family in tested]
    if not checked:
        return
    # El mateix a cada família d'una passada, el del seu information set.
    record = where / checked[0] / paths.FAMILY_RECORD
    gap = json.loads(record.read_text(encoding="utf-8"))["gap"]
    panels, axes = figure.panel_grid(
        len(checked), AUTOCORRELATION_PER_ROW, AUTOCORRELATION_ASPECT, sharey=True
    )
    for axis, family in zip(axes.flat, checked, strict=False):
        rows = stitched[stitched["family"] == family]
        lags = rows[rows["measure"] == "autocorrelation"]
        value = rows.loc[rows["measure"] == "ljung_box_p", "value"].item()
        days = rows.loc[rows["measure"] == "ljung_box_days", "value"].item()
        axis.bar(
            lags["subject"].astype(int),
            lags["value"].astype(float),
            color=colours[family],
            width=figure.BAR_WIDTH,
        )
        bound = WHITE_NOISE_Z / np.sqrt(days)
        axis.axhspan(
            -bound,
            bound,
            color=figure.REFERENCE_COLOUR,
            alpha=figure.BAND_OPACITY,
            linewidth=0,
        )
        for lag in (gap, DAYS_PER_WEEK):
            axis.axvline(lag, **figure.REFERENCE_LINE)
        axis.axhline(0, **figure.REFERENCE_LINE)
        axis.set_xlabel(figure.unit_label("retard", "dies"))
        axis.set_title(f"{figure.FAMILY_LABELS[family]}\n{figure.p_value(value)}")
    for axis in axes[:, 0]:
        axis.set_ylabel("autocorrelació")
    figure.save(panels, where, "autocorrelation")
