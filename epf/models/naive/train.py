"""Les tres prediccions ingènues de Lago et al. (2021), el terra que hi ha sota la referència.

No s'ajusta res: cada una copia les 24 hores d'un dia que ja es coneixia. Es copien de la
sèrie del preu i no de la matriu, perquè la setmana anterior no és cap entrada, i als dies
de prova que té la matriu diària, de manera que es puntuen sobre les files de totes les
famílies. La del dia similar porta, a més, la distribució amb què Marcjasz et al. (2023) la
fan probabilística. Es puntuen i no s'expliquen.
"""
import time

import numpy as np

from epf import paths
from epf.contract import matrix
from epf.contract.metrics import INTERVALS, scored, scored_distribution
from epf.contract.record import Fitted, save_model
from epf.contract.split import split
from epf.parameters import BLACKOUT_WEEK, DEVELOPMENT

PREVIOUS_DAY = 1
PREVIOUS_WEEK = 7

# Dies en què la predicció similar pren el dia anterior; la resta prenen la setmana
# anterior, perquè el cap de setmana no s'assembla als dies feiners. Dilluns és 0.
WEEKDAYS_FROM_YESTERDAY = (1, 2, 3, 4)

# Les parts de la llavor d'on la predicció ingènua pren els errors per fer-se
# probabilística: el que Marcjasz et al. (2023) diuen dins la mostra, l'entrenament i la
# validació, on la còpia no tria res.
IN_SAMPLE = ("training", "validation")


def _copied(price, days, lag):
    return price.shift(lag, freq="D").reindex(days).to_numpy(dtype=float)


def previous_day(price, days):
    """La mateixa hora del dia anterior, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_DAY)


def previous_week(price, days):
    """La mateixa hora de fa una setmana, `(dies, hores)`."""
    return _copied(price, days, PREVIOUS_WEEK)


def similar_day(price, days):
    """El dia anterior de dimarts a divendres, i la setmana anterior la resta."""
    yesterday = days.dayofweek.isin(WEEKDAYS_FROM_YESTERDAY)
    return np.where(yesterday[:, None], previous_day(price, days),
                    previous_week(price, days))


def bootstrapped(copied, errors, probabilities):
    """Els quantils de la còpia amb els seus errors dins la mostra, com Marcjasz et al.
    (2023) fan probabilística la predicció ingènua, `(dies, hores, probabilitats)`.

    L'article suma a la còpia els errors d'un dia sencer, trets a l'atzar amb reemplaçament.
    A cada hora això és la distribució dels errors d'aquella hora, i aquí se'n prenen els
    quantils exactes, la inversa de la seva distribució empírica, que és on van a parar
    els de les mostres com més n'hi ha. `copied` és `(dies, hores)` i `errors` `(dies de
    la mostra, hores)`.
    """
    spread = np.quantile(errors, probabilities, axis=0, method="inverted_cdf")
    return copied[:, :, None] + spread.T


# Cada predicció és una família, amb la funció que copia i els dies que copia, que és el
# que en diu el registre.
FORECASTS = {
    "naive_previous_day": (previous_day, {"lag": PREVIOUS_DAY}),
    "naive_previous_week": (previous_week, {"lag": PREVIOUS_WEEK}),
    "naive_similar_day": (similar_day, {"lag": PREVIOUS_DAY,
                                        "weekdays_from_yesterday": WEEKDAYS_FROM_YESTERDAY,
                                        "lag_otherwise": PREVIOUS_WEEK}),
}

# La que Marcjasz et al. (2023) fan probabilística, amb els errors de la mostra.
BOOTSTRAPPED = "naive_similar_day"

# Cada família, com la planifica una passada: els dies que copia, cap número de lloc ni
# cap configuració de cap cerca, perquè no se'n cerca cap.
FAMILIES = {name: {"fixed": numbers | ({"bootstrap_from": IN_SAMPLE}
                                       if name == BOOTSTRAPPED else {}),
                   "placeholder": None, "kept": 0, "searched_as": name}
            for name, (_, numbers) in FORECASTS.items()}


def _fitted(family, price, y, parts):
    """La còpia de la família als dies de prova de la matriu, i, de la del dia similar, la
    seva distribució puntuada, amb els errors dels dies de `IN_SAMPLE`, que es desen."""
    forecast, _ = FORECASTS[family]
    days = y.index[y.index.isin(parts.test)]
    copied = forecast(price, days)
    actual = y.loc[days]
    if family != BOOTSTRAPPED:
        return Fitted(copied.ravel(), actual.to_numpy().ravel(), {})
    sample = y.index[np.any([y.index.isin(getattr(parts, part)) for part in IN_SAMPLE],
                            axis=0)]
    errors = y.loc[sample].to_numpy() - forecast(price, sample)
    scores = scored_distribution(
        actual, lambda probabilities: bootstrapped(copied, errors, probabilities))
    return Fitted(copied.ravel(), actual.to_numpy().ravel(), {"errors": errors},
                  scores=scores)


def train(family, folder, configurations, explained_days):
    """La passada d'una família: per a cada llavor, la còpia puntuada i desada. Torna les
    parelles `[llavor, configuració]` que ha desat ara."""
    series = matrix.series()
    price, y = series[matrix.PRICE], matrix.matrices(series).actual
    done = []
    for seed, planned in configurations.items():
        parts = split(seed, DEVELOPMENT, BLACKOUT_WEEK)
        explained = parts.test if explained_days is None else parts.test[:explained_days]
        for configuration in planned:
            rank = configuration["rank"]
            path = folder / paths.SEED.format(seed=seed, rank=rank)
            if path.exists():
                continue
            start = time.time()
            fitted = _fitted(family, price, y, parts)
            scores = scored(fitted.predicted, fitted.actual) | (fitted.scores or {})
            save_model(path, parts, explained, fitted, scores, rank)
            done.append([seed, rank])
            distribution = "" if fitted.scores is None else (
                f", CRPS {scores['crps']:.2f}, hores que passen el test de Kupiec: "
                + " i ".join(f"{scores[f'kupiec_{level}']} al {level}%"
                             for level in INTERVALS))
            print(f"{family} llavor {seed} configuració {rank}: R² {scores['r2']:.3f}, "
                  f"MAE {scores['mae']:.2f} EUR/MWh{distribution}, "
                  f"{time.time() - start:.1f} s", flush=True)
    return done
