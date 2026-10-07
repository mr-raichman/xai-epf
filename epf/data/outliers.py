"""Com s'allunya cada valor dels seus veïns, i quins són els més allunyats.

La puntuació i l'informe que la llegeix viuen junts: la taula que s'imprimeix i la
figura que es dibuixa han de ser dels mateixos números. L'informe no refusa res i no
porta cap llindar, i no n'hi pot haver: la puntuació és distància partit per MAD, i
en aquesta sèrie els seus extrems són moviments ordinaris dins de finestres de preu
gairebé pla, no moviments grans.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from epf.market import HOURS_PER_DAY

# Veïns d'un preu: les hores adjacents de la sèrie cronològica, i la mateixa hora en
# els dies del voltant.
HOUR_HALF_WIDTH = 12
DAY_HALF_WIDTH = 7

# Quantes de les puntuacions més altes s'informen. És quantes files s'imprimeixen i
# no un llindar: no decideix res sobre cap valor.
HIGHEST = 100


def window_median_mad(values, half_width):
    """Mediana i MAD de la finestra centrada en cada valor, amb `half_width` valors
    a cada costat. Els extrems, sense finestra completa, valen NaN."""
    values = np.asarray(values, dtype=float)
    windows = sliding_window_view(values, 2 * half_width + 1)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    edge = np.full(half_width, np.nan)
    return np.concatenate([edge, median, edge]), np.concatenate([edge, mad, edge])


def hampel_score(values, half_width):
    """Distància a la mediana de la finestra centrada, en MADs de la finestra.

    Una finestra de MAD zero puntua 0, i els extrems sense finestra completa, NaN.
    """
    values = np.asarray(values, dtype=float)
    median, mad = window_median_mad(values, half_width)
    distance = np.abs(values - median)
    score = np.divide(distance, mad, out=np.zeros_like(distance), where=mad > 0)
    score[np.isnan(median)] = np.nan
    return score


def highest(scores):
    """Posicions de les `HIGHEST` puntuacions més altes, sense comptar les NaN."""
    flat = np.nan_to_num(scores.ravel(), nan=-np.inf)
    return np.argsort(flat)[-HIGHEST:]


def _describe(days, values, label, scores, median, mad):
    """Resum de les puntuacions més altes: com és la finestra i on cauen."""
    marks = highest(scores)
    years = pd.Series(days.repeat(HOURS_PER_DAY)[marks].year).value_counts().sort_index()
    print(f"{label}, les {HIGHEST} puntuacions més altes:")
    print(f"  MAD de la finestra, mediana {np.median(mad[marks]):.2f}, "
          f"màxim {mad[marks].max():.2f}")
    print(f"  mediana de la finestra, mediana {np.median(median[marks]):.2f}")
    print(f"  distància del valor a la mediana, mediana "
          f"{np.median(np.abs(values - median)[marks]):.2f}")
    print("  per any: " + ", ".join(f"{year} {count}" for year, count in years.items()))


def report(name, frame):
    """Imprimeix les puntuacions de veïnatge més altes i les torna, per si es dibuixen.

    Les dues vistes: contra les hores adjacents de la sèrie cronològica, i contra la
    mateixa hora dels dies del voltant.
    """
    values = frame.to_numpy().ravel()
    by_hours = hampel_score(values, HOUR_HALF_WIDTH).reshape(frame.shape)
    by_days = np.column_stack([hampel_score(frame[column].to_numpy(), DAY_HALF_WIDTH)
                               for column in frame.columns])

    hour_median, hour_mad = window_median_mad(values, HOUR_HALF_WIDTH)
    day_parts = [window_median_mad(frame[column].to_numpy(), DAY_HALF_WIDTH)
                 for column in frame.columns]
    print(f"veïnatge de {name}:")
    _describe(frame.index, values, f"hores adjacents (±{HOUR_HALF_WIDTH} h)", by_hours,
              hour_median, hour_mad)
    _describe(frame.index, values, f"mateixa hora (±{DAY_HALF_WIDTH} dies)", by_days,
              np.column_stack([part[0] for part in day_parts]).ravel(),
              np.column_stack([part[1] for part in day_parts]).ravel())
    return by_hours, by_days
