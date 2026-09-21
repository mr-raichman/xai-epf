"""Puntuació de com s'allunya cada valor dels seus veïns."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
