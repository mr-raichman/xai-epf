"""Puntuació de com s'allunya cada valor dels seus veïns."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def hampel_score(values, half_width):
    """Distància a la mediana de la finestra centrada, en MADs de la finestra.

    La finestra té `half_width` valors a cada costat. Una finestra de MAD zero
    puntua 0, i els valors dels extrems, sense finestra completa, NaN.
    """
    values = np.asarray(values, dtype=float)
    windows = sliding_window_view(values, 2 * half_width + 1)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    score = np.full(values.shape, np.nan)
    distance = np.abs(values[half_width:len(values) - half_width] - median)
    score[half_width:len(values) - half_width] = np.divide(
        distance, mad, out=np.zeros_like(distance), where=mad > 0)
    return score
