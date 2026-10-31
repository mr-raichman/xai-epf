"""La comprovació dels errors d'una passada, el test de Ljung-Box.

Diu si l'error de prova de cada família ajustada està correlacionat d'un dia a l'altre,
el test que Bergmeir, Hyndman i Koo (2018) fan a la validació creuada.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from epf import paths
from epf.contract import figure
from epf.parameters import DAYS_PER_WEEK, HOUR_COLUMNS
from epf.run import families

# Retards del test, dues vegades el període setmanal, com aconsellen Hyndman i
# Athanasopoulos (2021) per a una sèrie estacional.
LAGS = 2 * DAYS_PER_WEEK

# On cauen el 95 % de les autocorrelacions d'un soroll blanc: dins de ±1.96/√T, amb T la
# llargada de la sèrie, aquí els dies amb error (Hyndman i Athanasopoulos, 2021).
WHITE_NOISE_Z = 1.96

# Columnes de la figura, una família per panell, i l'alçada de cada panell entre la seva
# amplada.
AUTOCORRELATION_PER_ROW = 3
AUTOCORRELATION_ASPECT = 3 / 4

# El test de cada família, un per fila, al costat de la figura.
LJUNG_BOX = "ljung_box.csv"


def daily_errors(models):
    """Torna l'error mitjà de cada dia, del primer al darrer dia de prova.

    L'error és el preu menys la predicció. `models` són els models ajustats d'una
    família, en ordre de seed. Cada dia pren
    l'error del millor de la primera seed que el prova, i un dia que cap no prova val
    `NaN`. Falla si les files d'un fitxer no són les dels dies de prova que diu.
    """
    found = {}
    for model in models:
        if int(model["rank"]):
            continue
        days = model["days_test"]
        error = np.asarray(model["actual"]) - np.asarray(model["predicted"])
        if error.size != len(days) * len(HOUR_COLUMNS):
            raise ValueError(
                f"{error.size} errors per a {len(days)} dies de prova: les "
                f"files no són les dels dies que el fitxer diu"
            )
        for day, mean in zip(
            days, error.reshape(len(days), -1).mean(axis=1), strict=True
        ):
            found.setdefault(day, mean)
    span = np.arange(min(found), max(found) + np.timedelta64(1, "D"))
    return np.array([found.get(day, np.nan) for day in span])


def ljung_box(series, lags=LAGS):
    """Torna la Q de Ljung-Box, el seu valor p i les autocorrelacions d'una sèrie.

    Com `Box.test` de R, amb els dies buits passats a `acf`; sense cap dia buit és
    l'estimador de sempre. El valor p és el d'una khi quadrat de `lags` graus, sense
    restar-ne cap, perquè els errors són de fora de la mostra.
    """
    x = np.asarray(series, dtype=float) - np.nanmean(series)
    present = int((~np.isnan(x)).sum())
    covariances = []
    for lag in range(lags + 1):
        products = x[lag:] * x[: len(x) - lag]
        pairs = ~np.isnan(products)
        # Entre els parells més el retard, com `acf` de R, que sense cap dia buit és la
        # llargada de la sèrie.
        covariances.append(products[pairs].sum() / (pairs.sum() + lag))
    correlations = np.clip(np.array(covariances[1:]) / covariances[0], -1, 1)
    statistic = (
        present
        * (present + 2)
        * np.sum(correlations**2 / (present - np.arange(1, lags + 1)))
    )
    return statistic, stats.chi2.sf(statistic, lags), correlations


def report(where, described, colours):
    """Dibuixa el correlograma i desa el test dels errors de cada família que el fa.

    Les famílies són les de `described` el registre de les quals ho diu, a la passada
    `where`. El test va a `ljung_box.csv`, i el correlograma, a `autocorrelation.svg`,
    del color de cada família de `colours`; marca la distància de la partició i la
    setmana, i on cauria el 95 % de les autocorrelacions si l'error fos soroll blanc.
    """
    checked = []
    for family in families.EVERY:
        if family not in described:
            continue
        record = where / family / paths.FAMILY_RECORD
        made = json.loads(record.read_text(encoding="utf-8"))
        if made["ljung_box_tested"]:
            checked.append(family)
            # El mateix a cada família d'una passada, el del seu information set.
            gap = made["gap"]
    if not checked:
        return
    panels, axes = figure.panel_grid(
        len(checked), AUTOCORRELATION_PER_ROW, AUTOCORRELATION_ASPECT, sharey=True
    )
    tests = []
    for axis, family in zip(axes.flat, checked, strict=False):
        series = daily_errors(described[family])
        statistic, value, correlations = ljung_box(series)
        days = int((~np.isnan(series)).sum())
        tests.append(
            {"family": family, "lags": LAGS, "q": statistic, "p": value, "days": days}
        )
        axis.bar(
            np.arange(1, LAGS + 1),
            correlations,
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
    pd.DataFrame(tests).to_csv(
        where / LJUNG_BOX, index=False, lineterminator="\n", encoding="utf-8"
    )
