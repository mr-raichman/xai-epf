"""La comprovació dels residus d'una passada, el test de Ljung-Box.

Diu si l'error de prova de cada família ajustada està correlacionat d'un dia a l'altre,
el test que Bergmeir, Hyndman i Koo (2018) fan a la validació creuada.
"""

import numpy as np
import pandas as pd
from scipy import stats

from epf.contract import figure
from epf.contract.figure import plt
from epf.models.naive.train import FAMILIES as COPIES
from epf.parameters import DAYS_PER_WEEK, GAP, HOUR_COLUMNS
from epf.run import families

# Retards del test, dues vegades el període setmanal, com aconsellen Hyndman i
# Athanasopoulos (2021) per a una sèrie estacional.
LAGS = 2 * DAYS_PER_WEEK

# On cauen el 95 % de les autocorrelacions d'un soroll blanc: dins de ±1,96/√T, amb T la
# llargada de la sèrie, aquí els dies amb error (Hyndman i Athanasopoulos, 2021).
WHITE_NOISE_Z = 1.96

# Columnes de la figura, una família per panell, i l'alçada de cada panell entre la seva
# amplada.
PANELS_PER_ROW = 3
PANEL_ASPECT = 3 / 4

# El test de cada família, un per fila, al costat de la figura.
RESIDUALS = "residuals.csv"


def daily_errors(models):
    """Torna l'error mitjà de cada dia, del primer al darrer dia de prova.

    `models` són els models ajustats d'una família, en ordre de seed. Cada dia pren
    l'error del millor de la primera seed que el prova, i un dia que cap no prova val
    `NaN`. Falla si les files d'un fitxer no són les dels dies de prova que diu.
    """
    found = {}
    for model in models:
        if int(model["rank"]):
            continue
        days = model["days_test"]
        error = np.asarray(model["predicted"]) - np.asarray(model["actual"])
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
    restar-ne cap, perquè els residus són de fora de la mostra.
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
    """Dibuixa el correlograma de cada família ajustada i en desa el test.

    El test va a `residuals.csv`, i el correlograma, del color de cada família de
    `colours`, marca la distància de la partició i la setmana, i on cauria el 95 % de
    les autocorrelacions si l'error fos soroll blanc.
    """
    fitted = [
        family
        for family in families.EVERY
        if family in described and family not in COPIES
    ]
    if not fitted:
        return
    rows = -(-len(fitted) // PANELS_PER_ROW)
    panels, axes = plt.subplots(
        rows,
        PANELS_PER_ROW,
        figsize=figure.size(rows * figure.WIDTH / PANELS_PER_ROW * PANEL_ASPECT),
        squeeze=False,
        sharey=True,
    )
    tests = []
    for axis, family in zip(axes.flat, fitted, strict=False):
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
        for lag in (GAP, DAYS_PER_WEEK):
            axis.axvline(lag, **figure.REFERENCE_LINE)
        axis.axhline(0, **figure.REFERENCE_LINE)
        axis.set_xlabel("retard, dies")
        axis.set_title(f"{figure.FAMILY_LABELS[family]}\n{figure.p_value(value)}")
    for axis in axes[:, 0]:
        axis.set_ylabel("autocorrelació")
    for axis in axes.flat[len(fitted) :]:
        axis.set_visible(False)
    figure.save(panels, where, "residuals")
    pd.DataFrame(tests).to_csv(
        where / RESIDUALS, index=False, lineterminator="\n", encoding="utf-8"
    )
