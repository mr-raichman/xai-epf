"""Puntuació de com s'allunya cada valor dels seus veïns."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Mínim de la MAD, el pas de preu més petit. Amb una finestra plana, un valor
# diferent puntua molt alt en lloc de quedar sense puntuació.
MAD_FLOOR = 0.01


def hampel_score(values, half_width):
    """Distància a la mediana de la finestra centrada, en MADs de la finestra.

    La finestra té `half_width` valors a cada costat. Els valors dels extrems,
    sense finestra completa, puntuen NaN.
    """
    values = np.asarray(values, dtype=float)
    windows = sliding_window_view(values, 2 * half_width + 1)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    score = np.full(values.shape, np.nan)
    score[half_width:len(values) - half_width] = (
        np.abs(values[half_width:len(values) - half_width] - median) / np.maximum(mad, MAD_FLOOR))
    return score
