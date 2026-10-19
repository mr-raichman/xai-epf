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

# Retards del test, dues vegades el període setmanal, com aconsellen Hyndman i
# Athanasopoulos (2021) per a una sèrie estacional.
LAGS = 2 * DAYS_PER_WEEK

# Columnes de la figura, una família per quadre.
PANELS_PER_ROW = 4

# El test de cada família, un per fila, al costat de la figura.
RESIDUALS = "residuals.csv"


def daily_errors(models):
    """Torna l'error mitjà de cada dia, del primer al darrer dia de prova.

    `models` són els models ajustats d'una família, en ordre de llavor. Cada dia pren
    l'error del millor de la primera llavor que el prova, i un dia que cap no prova val
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


def report(where, described):
    """Dibuixa el correlograma de cada família ajustada i en desa el test.

    El test va a `residuals.csv`, i el correlograma marca la distància de la partició i
    la setmana.
    """
    fitted = [family for family in sorted(described) if family not in COPIES]
    if not fitted:
        return
    rows = -(-len(fitted) // PANELS_PER_ROW)
    panels, axes = plt.subplots(
        rows,
        PANELS_PER_ROW,
        figsize=(4 * PANELS_PER_ROW, 3 * rows),
        facecolor=figure.SURFACE,
        squeeze=False,
    )
    tests = []
    for axis, family in zip(axes.flat, fitted, strict=False):
        series = daily_errors(described[family])
        statistic, value, correlations = ljung_box(series)
        days = int((~np.isnan(series)).sum())
        tests.append(
            {"family": family, "lags": LAGS, "q": statistic, "p": value, "days": days}
        )
        axis.bar(np.arange(1, LAGS + 1), correlations, color=figure.SERIES, width=0.6)
        for lag in (GAP, DAYS_PER_WEEK):
            axis.axvline(lag, color=figure.MUTED_INK, linestyle="--", linewidth=0.8)
        axis.axhline(0, color=figure.INK, linewidth=0.6)
        axis.set_xlabel("retard, dies", color=figure.MUTED_INK)
        figure.style(axis, f"{family}, p {value:.2g}")
    for axis in axes.flat[len(fitted) :]:
        axis.set_visible(False)
    figure.save(panels, where, "residuals")
    pd.DataFrame(tests).to_csv(
        where / RESIDUALS, index=False, lineterminator="\n", encoding="utf-8"
    )
