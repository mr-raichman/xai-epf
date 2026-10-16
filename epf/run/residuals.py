"""La comprovació dels residus d'una passada: si l'error de prova de cada família
ajustada està correlacionat d'un dia a l'altre, el test que Bergmeir, Hyndman i Koo
(2018) fan a la validació creuada, el de Ljung-Box, calculat com el calcula R.

Les setmanes de prova de les llavors es cusen en una sola sèrie, cada dia de la primera
llavor que el prova i del millor model d'aquella llavor, i un dia que cap llavor no
prova queda buit. Un residu per dia, l'error mitjà de les seves hores. Les còpies
ingènues no s'ajusten, i la pregunta del test no és seva.
"""

import numpy as np
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


def daily_errors(models):
    """La sèrie de l'error mitjà de cada dia, del primer al darrer dia de prova, amb els
    dies de prova de cada llavor del seu millor model, el primer que el prova, i `NaN`
    on no en prova cap. `models` són els fitxers d'una família, en ordre de llavor."""
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
        for day, mean in zip(days, error.reshape(len(days), -1).mean(axis=1)):
            found.setdefault(day, mean)
    span = np.arange(min(found), max(found) + np.timedelta64(1, "D"))
    return np.array([found.get(day, np.nan) for day in span])


def ljung_box(series, lags=LAGS):
    """La Q de Ljung-Box, el seu valor p i les autocorrelacions, com `Box.test` de R amb
    els dies buits passats a `acf`.

    La sèrie es centra amb la mitjana dels dies presents; l'autocovariància de cada
    retard suma els parells on hi són tots dos dies i la divideix pel seu nombre més el
    retard; `n` són els dies presents; i `Q = n (n + 2) sum r_k^2 / (n - k)` es llegeix
    contra una khi quadrat de `lags` graus, sense restar-ne cap, perquè els residus són
    de fora de la mostra. Sense cap dia buit és l'estimador de sempre.
    """
    x = np.asarray(series, dtype=float) - np.nanmean(series)
    present = int((~np.isnan(x)).sum())
    covariances = []
    for lag in range(lags + 1):
        products = x[lag:] * x[: len(x) - lag]
        pairs = ~np.isnan(products)
        covariances.append(products[pairs].sum() / (pairs.sum() + lag))
    correlations = np.clip(np.array(covariances[1:]) / covariances[0], -1, 1)
    statistic = (
        present
        * (present + 2)
        * np.sum(correlations**2 / (present - np.arange(1, lags + 1)))
    )
    return statistic, stats.chi2.sf(statistic, lags), correlations


def report(where, described):
    """El correlograma de cada família ajustada, amb la distància de la partició i la
    setmana marcades, i una línia per família amb el test."""
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
    for axis, family in zip(axes.flat, fitted):
        series = daily_errors(described[family])
        statistic, value, correlations = ljung_box(series)
        days = int((~np.isnan(series)).sum())
        print(
            f"residus de {family}: Q de Ljung-Box a {LAGS} retards "
            f"{statistic:.1f}, valor p {value:.3g}, sobre {days} dies"
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
